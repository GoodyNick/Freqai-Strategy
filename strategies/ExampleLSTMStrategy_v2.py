import logging
from functools import reduce
from typing import Dict
import joblib
import os
from datetime import datetime

import numpy as np
import pandas as pd
import talib.abstract as ta
from technical import qtpylib

from pandas import DataFrame
from technical import qtpylib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from freqtrade import data
from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter
from freqtrade.persistence import Trade

logger = logging.getLogger(__name__)


class ExampleLSTMStrategy_v2(IStrategy):
    """
    This is an example strategy that uses the LSTMRegressor model to predict the target score.
    Use at your own risk.
    This is a simple example strategy and should be used for educational purposes only.
    """
    plot_config = {
        "main_plot": {},
        "subplots": {
            "predictions": {  
                "T": {"color": "blue", "plot_type": "line"},  # Model's Label
                "&-s_target": {"color": "red", "plot_type": "line"},  # Model's prediction
                "&-s_target_mean": {"color": "green", "plot_type": "line"},  # Model's prediction, average
                # "do_predict": {"color": "black", "plot_type": "scatter"},  # Show predictions as dots
            },
        },
    }
    
    # ROI table:
    minimal_roi = {
        "0": 1  # we let the model decide when to exit
    }

    # Stoploss:
    stoploss = -1  # Were letting the model decide when to sell

    # Trailing stop:
    trailing_stop = False
    trailing_stop_positive = 0.001
    trailing_stop_positive_offset = 0.0139
    trailing_only_offset_is_reached = True

    threshold_buy = RealParameter(-1, 1, default=0, space='buy')
    threshold_sell = RealParameter(-1, 1, default=0, space='sell')

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True
    use_custom_stoploss = True

    startup_candle_count = 20

    # Set to True to remove highly correlated features(enabling this causes 
    # mismatch between features in trained models and backtest features).
    # also, these should be enabled without using PCA
    do_remove_highly_correlated_features = False
    do_filter_important_features = True  
                                                
    prediction_metrics_storage = []  # Class-level storage for all pairs

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                    metadata: Dict, **kwargs):
        """
        Expands all features for FreqAI and applies filtering before training.
        """

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26
        )
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
            dataframe["bb_upperband-period"] - dataframe["bb_lowerband-period"]
        ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = dataframe["close"] / dataframe["bb_lowerband-period"]

        # ✅ Remove highly correlated features (if enabled)
        if self.do_remove_highly_correlated_features:
            logger.info(f"🔍 Removing highly correlated features.")
            dataframe = self.remove_highly_correlated_features(dataframe)

        # ✅ Apply feature filtering HERE
        if self.do_filter_important_features:
            dataframe = self.filter_important_features(dataframe)
        logger.info(f"🔍 Remaining Features After Filtering: {list(dataframe.columns)}")

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        # ✅ Assign `&-s_target` for FreqAI
        dataframe['&-s_target'] = self.create_target_T(dataframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.freqai_info = self.config["freqai"]

        dataframe['T'] = self.create_target_T(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['dynamic_stoploss'] = np.clip(-dataframe['atr'] * 1.75, -0.03, -0.10)

        dataframe = self.freqai.start(dataframe, metadata, self)          

        logger.info(f"🔍 do_predict distribution: {dataframe['do_predict'].value_counts()}")
        
        self.compute_prediction_metrics(dataframe, metadata)
        self.save_prediction_metrics()
        
        return dataframe


    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_threshold = 0.8  # Minimum confidence required to enter trades

        # ✅ Compute dynamic thresholds using rolling statistics of the prediction (`&-s_target`)
        prediction_mean = df["&-s_target"].rolling(100).mean()
        prediction_std = df["&-s_target"].rolling(100).std()

        long_threshold = prediction_mean + 0.5 * prediction_std  # ✅ Dynamic long threshold
        short_threshold = prediction_mean - 0.5 * prediction_std  # ✅ Dynamic short threshold

        # ✅ Use crossover conditions for trade triggers
        enter_long_conditions = [
            df["do_predict"] == 1,
            qtpylib.crossed_above(df["&-s_target"], long_threshold),  # ✅ Long trigger using crossover
            df["prediction_confidence"] > confidence_threshold,
            df["volume"] > 0
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            qtpylib.crossed_below(df["&-s_target"], short_threshold),  # ✅ Short trigger using crossover
            df["prediction_confidence"] > confidence_threshold,
            df["volume"] > 0
        ]

        # ✅ Apply entry conditions
        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_threshold = 0.85  # Slightly higher confidence for exits

        # ✅ Compute adaptive exit threshold based on rolling median of predictions (`&-s_target`)
        exit_threshold = df["&-s_target"].rolling(100).median()

        # ✅ Use crossovers instead of static comparisons
        strong_exit_long_conditions = [
            df["do_predict"] == 1,
            qtpylib.crossed_below(df["&-s_target"], exit_threshold),  # ✅ Exit long only on crossover
            df["prediction_confidence"] > confidence_threshold
        ]

        strong_exit_short_conditions = [
            df["do_predict"] == 1,
            qtpylib.crossed_above(df["&-s_target"], exit_threshold),  # ✅ Exit short only on crossover
            df["prediction_confidence"] > confidence_threshold
        ]

        # ✅ Apply exit conditions
        df.loc[reduce(lambda x, y: x & y, strong_exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "strong_exit_long")
        df.loc[reduce(lambda x, y: x & y, strong_exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "strong_exit_short")

        return df

    def create_target_T(self, dataframe: DataFrame):
        """
        Creates a target label (T) using scaled log future returns.
        """
        lookahead = 10  # Predicting 10 periods ahead

        # ✅ Compute log return target
        dataframe['future_return'] = np.log(dataframe['close'].shift(-lookahead) / dataframe['close'])

        # ✅ Scale `T` so that values are in a reasonable range for training
        dataframe['T'] = dataframe['future_return'].rolling(window=5).mean() * 100  # Multiply by 100

        # ✅ Optional: Normalize `T` (test with and without)
        # dataframe['T'] = np.tanh(dataframe['T'])

        # ✅ Fill NaNs
        dataframe['T'] = dataframe['T'].fillna(0)

        return dataframe['T']
    
    def custom_stoploss(self, pair, trade, current_time, current_rate, current_profit, **kwargs) -> float:
        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        
        if 'atr' not in df.columns:
            logger.warning(f"⚠️ ATR indicator missing for {pair}. Returning default stoploss.")
            return -0.05  # Default static stoploss
        
        atr_value = df.iloc[-1]['atr']
        trade_duration = (current_time - trade.open_date_utc).total_seconds() / 3600  # Convert to hours
        
        if current_profit > 0.02:
            atr_multiplier = max(1.8, current_profit * 5)  # ✅ Increase stop buffer with profit
        elif current_profit > 0:
            atr_multiplier = max(1.3, current_profit * 3)  # ✅ Protect small profits
        elif current_profit < -0.01:
            atr_multiplier = 0.8  # ❌ Tighten stop for losing trades
        else:
            atr_multiplier = 1.0
        
        # ✅ If trade is open for more than 24 hours, tighten stoploss
        if trade_duration > 24:
            atr_multiplier *= 0.7  # ✅ Reduce stoploss buffer by 30%
        
        if trade.is_short:
            stoploss_value = current_rate + (atr_value * atr_multiplier)
        else:
            stoploss_value = current_rate - (atr_value * atr_multiplier)
        
        return max(stoploss_value, trade.stop_loss)  # ✅ Ensures stoploss only tightens


    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, 
                        current_time, entry_tag, side: str, **kwargs) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1]

        confidence = last_candle["prediction_confidence"] if "prediction_confidence" in last_candle else 0.5

        # Adjust position size dynamically (scale trade size with confidence)
        adjusted_size = amount * confidence

        return super().confirm_trade_entry(pair, order_type, adjusted_size, rate, time_in_force, 
                                        current_time, entry_tag, side, **kwargs)

    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, label_col: str= "T", prediction_col: str = "&-s_target") -> pd.DataFrame: 
        """
        Computes and stores prediction accuracy metrics for all trading pairs.
        Saves the results to a CSV file after backtesting.
        """
        prediction_mean = prediction_col + "_mean"
        prediction_std = prediction_col + "_std"

        logger.info(f"🔍 {label_col} mean: {dataframe[label_col].mean()}, min: {dataframe[label_col].min()}, max: {dataframe[label_col].max()}")
        logger.info(f"🔍 {prediction_col} mean: {dataframe[prediction_col].mean()}, min: {dataframe[prediction_col].min()}, max: {dataframe[prediction_col].max()}")
        logger.info(f"🔍 {prediction_mean} mean: {dataframe[prediction_mean].mean()}, min: {dataframe[prediction_mean].min()}, max: {dataframe[prediction_mean].max()}")
        logger.info(f"🔍 {prediction_std} mean: {dataframe[prediction_std].mean()}, min: {dataframe[prediction_std].min()}, max: {dataframe[prediction_std].max()}")

        # Ensure required columns exist
        if prediction_col not in dataframe.columns:
            logger.warning(f"❌ Column '{prediction_col}' not found in dataframe. Skipping prediction metrics.")
            return dataframe

        # ✅ Step 1: Directional Accuracy (Sign Match)
        dataframe["prediction_correct"] = (np.sign(dataframe[label_col]) == np.sign(dataframe[prediction_col])).astype(int)

        # ✅ Step 2: Rolling Accuracy (Last 50 candles)
        dataframe["rolling_accuracy"] = dataframe["prediction_correct"].rolling(50, min_periods=1).mean()

        # ✅ Step 3: Mean Absolute Error (MAE)
        dataframe["mae"] = np.abs(dataframe[label_col] - dataframe[prediction_col]).rolling(100, min_periods=1).mean()

        # ✅ Step 4: Prediction Confidence (Normalized by Standard Deviation)
        std_col = prediction_std
        if std_col in dataframe.columns:
            dataframe["prediction_confidence"] = (np.abs(dataframe[prediction_col]) / (dataframe[std_col] + 1e-6)).clip(0, 1)

            # Confidence score is only counted for correct predictions
            dataframe["confidence_correct"] = np.where(
                dataframe["prediction_correct"] == 1, dataframe["prediction_confidence"], 0
            )

            # Normalize avg confidence over correct predictions
            correct_preds = dataframe["prediction_correct"].rolling(100, min_periods=1).sum()
            dataframe["avg_confidence_correct"] = dataframe["confidence_correct"].rolling(100, min_periods=1).sum() / (correct_preds + 1e-6)
        else:
            logger.warning(f"⚠️ Column '{std_col}' not found. Skipping confidence tracking.")
            dataframe["avg_confidence_correct"] = np.nan

        # ✅ Step 5: Calculate Fraction of Predicted Targets
        total_predictions = (dataframe["do_predict"] == 1).sum()
        logger.info(f"🔍 `do_predict=1` Count: {total_predictions}, `do_predict=-1` Count: {(dataframe['do_predict'] == -1).sum()}")
        total_targets_available = dataframe[label_col].notna().sum()
        fraction_predicted = total_predictions / total_targets_available if total_targets_available > 0 else 0

        # ✅ Step 6: Store Metrics in Class-Level List
        pair = metadata["pair"]
        metrics = {
            "pair": pair,
            "total_predictions": total_predictions,
            "fraction_predicted": fraction_predicted,
            "rolling_accuracy": dataframe["rolling_accuracy"].iloc[-1],
            "mae": dataframe["mae"].iloc[-1],
            "avg_confidence_correct": dataframe["avg_confidence_correct"].iloc[-1] if "avg_confidence_correct" in dataframe.columns else np.nan,
            "correlation": dataframe[prediction_col].corr(dataframe[label_col])  # ✅ Step 8: Correlation between Target and Predictions
        }
        self.prediction_metrics_storage.append(metrics)

        # ✅ Step 7: Log Key Statistics
        logger.info(
            "🔍 Prediction Metrics | Pair: %s | Total Predictions: %s | Fraction Predicted: %.4f | Rolling Accuracy: %.4f | MAE: %.6f | Avg Confidence: %.4f | Correlation: %.4f",
            pair, total_predictions, fraction_predicted, metrics["rolling_accuracy"], metrics["mae"], metrics["avg_confidence_correct"], metrics["correlation"]
        )

        return dataframe

    def save_prediction_metrics(self, filename="prediction_metrics.csv"):
        """
        Saves the accumulated prediction metrics to a CSV file after backtesting.
        """
        if not self.prediction_metrics_storage:
            logger.warning("⚠️ No prediction metrics found to save.")
            return

        df = pd.DataFrame(self.prediction_metrics_storage)
        output_path = os.path.join(self.config["user_data_dir"], filename)
        df.to_csv(output_path, index=False)

        logger.info(f"✅ Prediction metrics saved to {output_path}")

    def remove_highly_correlated_features(self, dataframe, threshold=0.85):
        """
        Drops one feature from each highly correlated pair, except essential features.
        """
        essential_features = {"high", "low", "close", "open", "volume"}  # Features FreqAI depends on

        corr_matrix = dataframe.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))

        to_drop = [column for column in upper.columns if any(upper[column] > threshold) and column not in essential_features]

        if to_drop:
            logger.info(f"Dropping {len(to_drop)} highly correlated features: {to_drop}")
            dataframe = dataframe.drop(columns=to_drop)

        return dataframe 
          
    def filter_important_features(self, dataframe):
        """
        Removes all columns that start with '%' unless they are in the important features list.
        """
        important_features = {
            "%-hour_of_day",
            "%-day_of_week",
            "%-cci-period_50_BTC/USDTUSDT_4h",
            "%-pct-change_gen_BTC/USDTUSDT_1h",
            "%-roc-period_20_BTC/USDTUSDT_2h",
            "%-rsi-period_10_BTC/USDTUSDT_4h",
            "%-rsi-period_50_ETH/USDTUSDT_4h",
            "%-bb_width-period_50_BTC/USDTUSDT_4h"
        }

        # Drop all columns starting with '%' unless they are in the important_features set
        columns_to_keep = [col for col in dataframe.columns if not col.startswith("%") or col in important_features]
        
        return dataframe[columns_to_keep]

