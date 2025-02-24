import logging
import torch
import numpy as np
import random

# ✅ Ensure full determinism across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Use the same seed every time

from typing import Dict, Any

import os
import pandas as pd
from joblib import Parallel, delayed

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchLSTMTrainer

logger = logging.getLogger(__name__)

class PyTorchLSTMRegressor_v1(BasePyTorchRegressor):
    """
    PyTorchLSTMRegressor is a class that uses a PyTorch LSTM model to predict a continuous target variable.
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, **kwargs) -> None:
        super().__init__(**kwargs)
        config = self.freqai_info.get("model_training_parameters", {})
        self.learning_rate: float = config.get("learning_rate", 3e-4)
        self.model_kwargs: Dict[str, Any] = config.get("model_kwargs", {})
        self.trainer_kwargs: Dict[str, Any] = config.get("trainer_kwargs", {})
        self.window_size = self.model_kwargs.get('window_size', 10)

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train the LSTM model using provided data."""

        # Check if GPU should be used based on config
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("🚀 Using GPU for training!")
        else:
            self.device = torch.device("cpu")
            logger.info("🖥️ Using CPU for training.")

        n_features = data_dictionary["train_features"].shape[-1]
        model = PyTorchLSTMModel(input_dim=n_features, output_dim=1, **self.model_kwargs)
        model.to(self.device)

        # ✅ Use SGD instead of AdamW for deterministic training
        optimizer = torch.optim.SGD(model.parameters(), lr=self.learning_rate, momentum=0.9)
        criterion = torch.nn.MSELoss(reduction='mean')

        trainer = self.get_init_model(dk.pair)
        
        if trainer is None:
            trainer = PyTorchLSTMTrainer(
                model=model,
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                window_size=self.window_size,
                **self.trainer_kwargs,
            )

        trainer.fit(data_dictionary, self.splits)
        self.model = trainer

        return trainer
