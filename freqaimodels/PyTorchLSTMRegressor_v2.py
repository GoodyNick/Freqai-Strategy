import logging
import torch
import numpy as np
import random
import os
import pandas as pd
from typing import Dict, Any

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel_v2 import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer_v2 import PyTorchLSTMTrainer

logger = logging.getLogger(__name__)

# ✅ Ensure full determinism across runs
def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)  # Use the same seed every time

class PyTorchLSTMRegressor_v2(BasePyTorchRegressor):
    """
    PyTorchLSTMRegressor is a class that uses a PyTorch LSTM model to predict a continuous target variable.
    """

    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, model_training_parameters=None, model_kwargs=None, config=None):
        # ✅ Ensure parameters are not None
        model_training_parameters = model_training_parameters or {}
        model_kwargs = model_kwargs or {}

        # ✅ Pass config correctly
        super().__init__(config=config)

        # ✅ Load model hyperparameters
        self.hidden_dim = model_kwargs.get("hidden_dim", 128)  
        self.window_size = model_kwargs.get("window_size", 30)  
        self.num_layers = model_kwargs.get("num_lstm_layers", 3)  
        self.dropout = model_kwargs.get("dropout_percent", 0.2)  

        # ✅ Modify optimizer and training settings
        self.lr = model_training_parameters.get("learning_rate", 0.0005)
        self.weight_decay = model_training_parameters.get("weight_decay", 0.00005)
        self.num_epochs = model_training_parameters.get("num_epochs", 50)

        # ✅ Adjust batch size for stable training
        self.batch_size = model_training_parameters.get("trainer_kwargs", {}).get("batch_size", 64)

        # ✅ Ensure `trainer_kwargs` is assigned properly
        self.trainer_kwargs = model_training_parameters.get("trainer_kwargs", {})

        # ✅ Initialize model later, after input feature count is determined
        self.model = None

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        """Train the LSTM model using provided data."""

        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)

        if use_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda")
            logger.info("🚀 Using GPU for training!")
        else:
            self.device = torch.device("cpu")
            logger.info("🖥️ Using CPU for training.")

        # ✅ Get actual feature count from training data
        n_features = data_dictionary["train_features"].shape[-1]
        logger.info(f"🔍 Detected {n_features} features for LSTM input.")

        # ✅ Ensure `self.model` is correctly initialized
        if self.model is None:
            self.model = PyTorchLSTMModel(
                input_dim=n_features,  # ✅ Use actual feature count
                output_dim=1,
                num_layers=self.num_layers,
                dropout=self.dropout
            ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=0.001, weight_decay=0.0001)
        criterion = torch.nn.SmoothL1Loss()  # Huber Loss

        trainer = self.get_init_model(dk.pair)

        if trainer is None:
            trainer = PyTorchLSTMTrainer(
                model=self.model,  # ✅ Use `self.model`
                optimizer=optimizer,
                criterion=criterion,
                device=self.device,
                data_convertor=self.data_convertor,
                tb_logger=self.tb_logger,
                window_size=self.window_size,
                **self.trainer_kwargs,
            )

        # ✅ Ensure `self.model` remains the same instance
        self.model = trainer.model

        trainer.fit(data_dictionary, self.splits)

        # ✅ Compute feature importance if enabled
        if self.config["freqai"]["model_training_parameters"].get("enable_feature_importance", False):
            self.compute_feature_importance(data_dictionary)

        return trainer

    def compute_feature_importance(self, data_dictionary: Dict[str, pd.DataFrame], save_path="feature_importances.csv"):
        """
        Compute feature importance scores based on absolute weight magnitudes.
        Saves results to a CSV file.
        """
        feature_names = data_dictionary["train_features"].columns.tolist()

        # Extract model weights from the first LSTM layer
        with torch.no_grad():
            first_layer_weights = self.model.lstm_layers[0].weight_ih_l0.abs().sum(dim=0).cpu().numpy()

        # Normalize importance scores
        importance_scores = first_layer_weights / first_layer_weights.sum()

        # Store as DataFrame and save
        df = pd.DataFrame({"Feature": feature_names, "Importance": importance_scores})
        df = df.sort_values(by="Importance", ascending=False)

        output_path = os.path.join(self.config["user_data_dir"], save_path)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Feature importance scores saved to {output_path}")
