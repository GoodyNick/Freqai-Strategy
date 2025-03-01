import logging
import torch
import numpy as np
import random
import os
import pandas as pd
from typing import Dict, Any, Tuple

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel_v2 import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer_v2 import PyTorchLSTMTrainer

logger = logging.getLogger(__name__)

def set_seed(seed=42):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

class PyTorchLSTMRegressor_v2(BasePyTorchRegressor):
    @property
    def data_convertor(self) -> PyTorchDataConvertor:
        return DefaultPyTorchDataConvertor(target_tensor_type=torch.float)

    def __init__(self, model_training_parameters=None, model_kwargs=None, config=None):
        model_training_parameters = model_training_parameters or {}
        model_kwargs = model_kwargs or {}
        super().__init__(config=config)

        self.hidden_dim = model_kwargs.get("hidden_dim", None)  # ✅ Avoids conflict with feature count
        self.window_size = model_kwargs.get("window_size", config["freqai"]["model_kwargs"].get("window_size", 24))
        self.num_layers = model_kwargs.get("num_lstm_layers", config["freqai"]["model_kwargs"].get("num_lstm_layers", 3))
        self.dropout = model_kwargs.get("dropout_percent", config["freqai"]["model_kwargs"].get("dropout_percent", 0.2))

        self.lr = model_training_parameters.get("learning_rate", config["freqai"]["model_training_parameters"].get("learning_rate", 0.0005))
        self.weight_decay = model_training_parameters.get("weight_decay", config["freqai"]["model_training_parameters"].get("weight_decay", 0.00005))
        self.num_epochs = model_training_parameters.get("num_epochs", config["freqai"]["model_training_parameters"].get("num_epochs", 50))

        self.batch_size = model_training_parameters.get("trainer_kwargs", {}).get("batch_size", config["freqai"]["model_training_parameters"]["trainer_kwargs"].get("batch_size", 64))
        self.trainer_kwargs = model_training_parameters.get("trainer_kwargs", config["freqai"]["model_training_parameters"].get("trainer_kwargs", {}))

        self.model = None

    def fit(self, data_dictionary: Dict, dk: FreqaiDataKitchen, **kwargs) -> Any:
        use_gpu = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("use_gpu", False)
        self.device = torch.device("cuda" if use_gpu and torch.cuda.is_available() else "cpu")
        logger.info(f"🚀 Using {'GPU' if self.device.type == 'cuda' else 'CPU'} for training!")

        # ✅ Read `reshape_3d` from config
        reshape_3d = dk.config.get("freqai", {}).get("feature_parameters", {}).get("reshape_3d", [24, 150])
        self.window_size, n_features = reshape_3d  # ✅ Dynamically set time_steps and feature count

        num_samples = data_dictionary["train_features"].shape[0]

        logger.info(f"🔍 Expected input size: {n_features} features, Time Steps: {self.window_size}")

        if num_samples < self.window_size:
            raise ValueError(f"🚨 Not enough samples ({num_samples}) for the required sequence length ({self.window_size}).")

        if num_samples % self.window_size != 0:
            new_size = (num_samples // self.window_size) * self.window_size
            logger.warning(f"⚠ Reshaping issue detected! Adjusting samples from {num_samples} to {new_size} for LSTM sequence formatting.")
            data_dictionary["train_features"] = data_dictionary["train_features"][:new_size]
            data_dictionary["train_labels"] = data_dictionary["train_labels"][:new_size]

        self.trained_feature_names = list(data_dictionary["train_features"].columns)

        train_features_np = data_dictionary["train_features"].values
        train_labels_np = data_dictionary["train_labels"].values

        # ✅ Reshape features manually
        num_batches = train_features_np.shape[0] // self.window_size
        train_features_np = train_features_np.reshape(num_batches, self.window_size, n_features)

        # 🔥 Fix: Reshape labels correctly to match sequence format
        train_labels_np = train_labels_np.reshape(num_batches, self.window_size, 1)  # ✅ Ensure labels match sequences

        logger.info(f"✅ Feature dimensions after manual reshaping: {train_features_np.shape}")
        logger.info(f"✅ Label dimensions after manual reshaping: {train_labels_np.shape}")

        # Convert back to Pandas DataFrame before saving
        data_dictionary["train_features"] = pd.DataFrame(
            train_features_np.reshape(-1, n_features),
            columns=self.trained_feature_names
        )
        data_dictionary["train_labels"] = pd.DataFrame(
            train_labels_np.reshape(-1, 1),
            columns=["T"]
        )

        # Convert to PyTorch tensors for model training
        train_features_tensor = torch.tensor(train_features_np, dtype=torch.float32)
        train_labels_tensor = torch.tensor(train_labels_np, dtype=torch.float32)

        logger.info(f"✅ Final Feature dimensions before training: {train_features_tensor.shape}")
        logger.info(f"✅ Final Label dimensions before training: {train_labels_tensor.shape}")

        self.model = PyTorchLSTMModel(
            input_dim=n_features,
            output_dim=1,
            num_layers=self.num_layers,
            dropout=self.dropout
        ).to(self.device)

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        criterion = torch.nn.SmoothL1Loss()
        trainer = PyTorchLSTMTrainer(
            model=self.model,
            optimizer=optimizer,
            criterion=criterion,
            device=self.device,
            data_convertor=self.data_convertor,
            tb_logger=self.tb_logger,
            window_size=self.window_size,
            **self.trainer_kwargs,
        )

        try:
            trainer.fit(data_dictionary, self.splits)
            self.model = trainer.model

        except Exception as e:
            logger.error(f"🚨 Training failed with error: {e}")
            self.model = None
            raise e

        return self.model

    def predict(self, data_dictionary: Dict, dk=None) -> Tuple[pd.DataFrame, bool]:
        """
        Generate predictions using the trained PyTorch LSTM model.

        :param data_dictionary: Dictionary containing feature data.
        :param dk: (Optional) Data kitchen object passed by FreqAI.
        :returns: Tuple (predictions_df, do_predict) where:
                    - predictions_df: pandas DataFrame with predictions
                    - do_predict: Boolean flag indicating if predictions were made
        """

        if not self.model:
            raise ValueError("🚨 Model is not properly loaded. Ensure training has been completed.")

        expected_features = set(self.trained_feature_names)  # ✅ Features used during training
        available_features = set(data_dictionary.keys())  # ✅ Features available for prediction

        missing_features = expected_features - available_features
        if missing_features:
            logger.error(f"🚨 Missing features in data_dictionary! Missing: {missing_features}")
            raise KeyError(f"🚨 Required features for prediction are missing! Ensure the dataframe includes: {expected_features}")

        logger.info(f"🔍 Predicting with {len(available_features)} available features.")

        # ✅ Convert dictionary keys to DataFrame
        features_df = pd.DataFrame(data_dictionary)

        # ✅ Ensure correct feature ordering
        features_df = features_df[self.trained_feature_names]  # ✅ Ensuring correct feature selection

        # Convert dataframe to tensor
        features_tensor = torch.tensor(features_df.values, dtype=torch.float32).to(self.device)

        # Perform prediction
        self.model.eval()  # ✅ Ensure model is in evaluation mode
        with torch.no_grad():
            predictions = self.model(features_tensor).cpu().numpy()

        # ✅ Convert NumPy array to DataFrame
        predictions_df = pd.DataFrame(predictions, columns=["&-s_target"])

        logger.info(f"✅ Predictions Shape: {predictions_df.shape}")

        return predictions_df, True  # ✅ Returning a DataFrame instead of NumPy array

    def format_predictions_for_inverse_transform(self, predictions):
        if isinstance(predictions, torch.Tensor):
            predictions = predictions.cpu().numpy()

        if predictions.ndim == 3:
            predictions = predictions.squeeze(-1)

        if predictions.ndim == 2:
            predictions = predictions[:, -1]

        return predictions  # ✅ Return correctly formatted predictions
