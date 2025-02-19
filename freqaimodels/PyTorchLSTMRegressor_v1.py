from typing import Dict, Any

import torch
import os
import numpy as np
import pandas as pd

from freqtrade.freqai.base_models.BasePyTorchRegressor import BasePyTorchRegressor
from freqtrade.freqai.data_kitchen import FreqaiDataKitchen
from freqtrade.freqai.torch.PyTorchDataConvertor import PyTorchDataConvertor, DefaultPyTorchDataConvertor
from freqtrade.freqai.torch.PyTorchLSTMModel import PyTorchLSTMModel
from freqtrade.freqai.torch.PyTorchModelTrainer import PyTorchLSTMTrainer

import logging
logger = logging.getLogger(__name__)

class PyTorchLSTMRegressor_v1(BasePyTorchRegressor):
    """
    PyTorchLSTMRegressor is a class that uses a PyTorch LSTM model to predict a continuous target variable.

      "model_training_parameters": {
      "learning_rate": 3e-3,
      "trainer_kwargs": {
        "n_steps": null,
        "batch_size": 32,
        "n_epochs": 10,
      },
      "model_kwargs": {
        "num_lstm_layers": 3,
        "hidden_dim": 128,
        "window_size": 5,
        "dropout_percent": 0.4
      }
    }
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

        optimizer = torch.optim.AdamW(model.parameters(), lr=self.learning_rate)
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

        # ✅ Check if feature importance is enabled in config
        enable_feature_importance = dk.config.get("freqai", {}).get("model_training_parameters", {}).get("enable_feature_importance", False)

        # ✅ Compute Feature Importance Only if Enabled
        if enable_feature_importance:
            self.permutation_importance(
                self.model.model,
                data_dictionary["train_features"],
                data_dictionary["train_labels"],
                pair_name=dk.pair  # ✅ Ensure this is correctly passed
            )
            logger.info(f"✅ Feature importance scores successfully saved to feature_importances.csv")

        return trainer

    def permutation_importance(self, model, X_test, y_test, pair_name, feature_names=None, 
                            importances_file="./user_data/feature_importances.csv", metric='mae', 
                            num_shuffles=5, top_n=100, importance_threshold=None):
        """
        Compute Permutation Importance for a trained LSTM model and append results to feature_importances.csv.
        """
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        model.to(device)
        model.eval()

        # ✅ Debug: Check if pair_name is correctly passed
        logger.info(f"📌 Calculating feature importance scores for Pair: {pair_name}")

        # ✅ Extract feature names if X_test is a DataFrame
        if isinstance(X_test, pd.DataFrame):
            feature_names = feature_names or X_test.columns.tolist()
            X_test = torch.tensor(X_test.values, dtype=torch.float32)
        elif feature_names is None:
            feature_names = [f"Feature_{i}" for i in range(X_test.shape[1])]  # Fallback if no names

        if isinstance(y_test, pd.DataFrame) or isinstance(y_test, pd.Series):
            y_test = y_test.to_numpy()  # ✅ Convert y_test to NumPy array

        X_test = X_test.to(device)

        with torch.no_grad():
            y_pred_baseline = model(X_test).cpu().numpy()

        def calc_metric(y_true, y_pred):
            if metric == 'mae':
                return np.mean(np.abs(y_true - y_pred))
            elif metric == 'mse':
                return np.mean((y_true - y_pred) ** 2)
            else:
                raise ValueError("Unsupported metric. Use 'mae' or 'mse'.")

        baseline_score = calc_metric(y_test, y_pred_baseline)

        feature_importance = []

        for feature_idx in range(X_test.shape[1]):
            scores = []
            for _ in range(num_shuffles):
                X_shuffled = X_test.clone()
                perm = torch.randperm(X_test.shape[0])
                X_shuffled[:, feature_idx] = X_shuffled[perm, feature_idx]

                with torch.no_grad():
                    y_pred_permuted = model(X_shuffled).cpu().numpy()

                score = calc_metric(y_test, y_pred_permuted)
                scores.append(score)

            importance_value = np.mean(scores) - baseline_score
            feature_importance.append((pair_name, feature_names[feature_idx], importance_value))

        # ✅ Create DataFrame with Proper Feature Names & Sorting
        df_importance = pd.DataFrame(feature_importance, columns=["Pair", "Feature", "Importance"])

        # ✅ Debug: Check if the Pair column exists in the DataFrame
        if "Pair" not in df_importance.columns:
            logger.warning("⚠️ 'Pair' column is missing from DataFrame before saving!")

        df_importance = df_importance.sort_values(by="Importance", ascending=False)  # ✅ Sort Descending

        # ✅ Apply Filtering Options
        if top_n:
            df_importance = df_importance.head(top_n)
        if importance_threshold:
            df_importance = df_importance[df_importance["Importance"] >= importance_threshold]

        # ✅ Check if file exists; append if it does, else write normally
        if os.path.exists(importances_file):
            df_importance.to_csv(importances_file, mode="a", header=False, index=False, columns=["Pair", "Feature", "Importance"])
        else:
            df_importance.to_csv(importances_file, mode="w", header=True, index=False, columns=["Pair", "Feature", "Importance"])
        





