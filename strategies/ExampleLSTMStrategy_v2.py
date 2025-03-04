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
from scipy.fftpack import fft
from scipy.stats import zscore
from torch import mul

from freqtrade import data
from freqtrade.exchange.exchange_utils import *
from freqtrade.optimize.analysis import lookahead
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
        "main_plot": {
        },
        "subplots": {
            "predictions": {
                "True Label": {"color": "blue", "plot_type": "line"},  # Rename T to "True Label"
                "Prediction": {"color": "red", "plot_type": "line"},  # Rename "&-s_target" to "Prediction"
                "Avg Prediction": {"color": "green", "plot_type": "line"},  # Rename "&-s_target_mean" to "Avg Prediction"
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
    do_filter_important_features = False  
                                                
    prediction_metrics_storage = []  # Class-level storage for all pairs

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, period: int, metadata: Dict, **kwargs):
        """
        Expands features that benefit from multiple timeframes.
        """

        # ✅ Momentum & Trend Indicators (Expanded Over Timeframes)
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe["%-ma-period"] = ta.SMA(dataframe, timeperiod=10)
        dataframe["%-roc-period"] = ta.ROC(dataframe, timeperiod=2)
        
        # ✅ MACD
        dataframe["%-macd-period"], dataframe["%-macdsignal-period"], dataframe["%-macdhist-period"] = ta.MACD(
            dataframe['close'], slowperiod=12, fastperiod=26
        )

        # ✅ Bollinger Bands
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

        # ✅ Fix NaNs in Expanded Features
        expanded_features = [
            "%-cci-period", "%-rsi-period", "%-momentum-period", "%-ma-period",
            "%-roc-period", "%-macd-period", "%-macdsignal-period", "%-macdhist-period",
            "bb_lowerband-period", "bb_upperband-period", "%-bb_width-period", "%-close-bb_lower-period"
        ]
        dataframe[expanded_features] = dataframe[expanded_features].bfill().fillna(0)  # ✅ Fix applied

        if self.do_remove_highly_correlated_features:
            dataframe = self.remove_highly_correlated_features(dataframe)

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        return dataframe

    def feature_engineering_standard(self, dataframe: pd.DataFrame, metadata: Dict, **kwargs):
        """
        Defines features that should remain in their original timeframe.
        """

        # ✅ Keep existing time-based features
        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe.loc[:, "%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe.loc[:, "%-hour_of_day"] = dataframe["date"].dt.hour

        # ✅ Rolling Features (Fixed NaNs)
        dataframe.loc[:, "%-rolling_volatility"] = dataframe["close"].rolling(window=24).std().bfill()
        dataframe.loc[:, "%-rolling_mean"] = dataframe["close"].rolling(window=24).mean().bfill()

        # ✅ CUSUM (Trend Break Detector - Should NOT be expanded)
        def get_cusum(series):
            series_mean = series.mean()
            return (series - series_mean).cumsum()

        dataframe.loc[:, "%-cusum_close"] = get_cusum(dataframe["close"]).fillna(0)

        # ✅ Hurst Exponent (Trend Strength - Fixed NaNs)
        def hurst_exponent(ts, max_lag=20):
            if len(ts) < max_lag:
                return np.nan
            lags = range(2, max_lag)
            tau = [np.std(np.subtract(ts[lag:], ts[:-lag])) for lag in lags]
            return np.polyfit(np.log(lags), np.log(tau), 1)[0]

        dataframe.loc[:, "%-hurst"] = dataframe["close"].rolling(window=72).apply(hurst_exponent, raw=True)
        dataframe.loc[:, "%-hurst"] = dataframe["%-hurst"].fillna(dataframe["%-hurst"].median())

        # ✅ Fourier Transform (Fixed NaNs)
        def compute_fourier(series, n_components=3):
            if len(series) < 72:
                return np.nan
            fft_vals = fft(series)
            return np.abs(fft_vals[:n_components]).sum()

        dataframe.loc[:, "%-fourier_price"] = dataframe["close"].rolling(window=72).apply(compute_fourier, raw=True)
        dataframe.loc[:, "%-fourier_price"] = dataframe["%-fourier_price"].fillna(dataframe["%-fourier_price"].median())

        # ✅ Fix Z-Score Normalization NaNs (Apply AFTER filling raw features)
        zscore_columns = ["%-rolling_volatility", "%-rolling_mean", "%-cusum_close", "%-hurst", "%-fourier_price"]
        for col in zscore_columns:
            dataframe.loc[:, f"{col}-zscore"] = pd.Series(zscore(dataframe[col]), index=dataframe.index).fillna(0)  # ✅ Convert to Series

        logger.info(f"🔍 Total features before model training: {len(dataframe.columns)}")

        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        # ✅ Assign `&-s_target` for FreqAI
        dataframe['&-s_target'] = self.create_target_T(dataframe)

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:
        self.freqai_info = self.config["freqai"]

        # this is to be used for plotting and stoploss
        dataframe['T'] = self.create_target_T(dataframe)
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        # Compute ATR percentiles (Dynamic Scaling Without Fixed Factors)
        atr_percentile_25 = dataframe['atr'].rolling(100).quantile(0.25)
        atr_percentile_75 = dataframe['atr'].rolling(100).quantile(0.75)        
        # Define a dynamic multiplier based on ATR position within its range
        dataframe["dynamic_multiplier"] = 1.0 + ((dataframe["atr"] - atr_percentile_25) / (atr_percentile_75 - atr_percentile_25)).clip(0, 1)
        
        logger.info(f"🔍 Feature dimensions before training: {dataframe.shape}")

        dataframe = self.freqai.start(dataframe, metadata, self)          

        logger.info(f"🔍 do_predict distribution: {dataframe['do_predict'].value_counts()}")

        # Compute thresholds using fully dynamic multiplier
        dataframe["long_threshold"] = dataframe["&-s_target_mean"] + dataframe["&-s_target_std"] * dataframe["dynamic_multiplier"]
        dataframe["short_threshold"] = dataframe["&-s_target_mean"] - dataframe["&-s_target_std"] * dataframe["dynamic_multiplier"]

        """
        Adds aliases for plotting while keeping the original columns intact.
        """
        dataframe["Prediction"] = dataframe["&-s_target"]
        dataframe["Avg Prediction"] = dataframe["&-s_target_mean"]
        dataframe["True Label"] = dataframe["T"]

        self.compute_prediction_metrics(dataframe, metadata)
        self.save_prediction_metrics()
        
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_threshold = 0.45  # Reduced to allow more trades

        # ✅ Adaptive Trend Thresholds
        df["dynamic_T_threshold"] = df["atr"] * 0.002
        df["vol_rank"] = df["volume"].rolling(50).rank(pct=True)
        df["valid_volume"] = df["vol_rank"] > 0.10  # More permissive

        enter_long_conditions = [
            df["do_predict"] == 1,  
            df["T"] > 0,  # ✅ Allow weaker bullish trends (was filtering small trends before)
            df["valid_volume"] == True,  
            df["prediction_confidence"] > confidence_threshold
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,  
            df["T"] < -df["dynamic_T_threshold"],
            df["valid_volume"] == True,
            df["prediction_confidence"] > confidence_threshold
        ]

        df.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        df.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_threshold = 0.45  # Reduced to allow more exits

        # ✅ Compute dynamic exit threshold
        df["dynamic_exit_threshold"] = df["&-s_target"].rolling(2).mean() + (df["atr"] * 0.0008)
        
        # ✅ Faster trend change detection
        df["trend_change"] = (df["T"].diff(1).abs() > df["T"].rolling(2).std())

        # ✅ Ensure profit-based exit exists
        if "current_profit" in df.columns:
            df["profit_based_exit"] = (df["current_profit"] > 0.005) & (df["T"] < 0.005)  # ✅ Reduced from 1.5%
        else:
            df["profit_based_exit"] = False  

        strong_exit_long_conditions = [
            df["do_predict"] >= 0,
            df["trend_change"] == True,  
            df["&-s_target"] < df["dynamic_exit_threshold"],  
            (df["T"] < 0.01) | df["profit_based_exit"],  
            df["prediction_confidence"] > confidence_threshold
        ]

        strong_exit_short_conditions = [
            df["do_predict"] >= 0,
            df["trend_change"] == True,  
            df["&-s_target"] > df["dynamic_exit_threshold"],
            (df["T"] > -0.05) | df["profit_based_exit"],  # ✅ Lowered from -0.03
            df["prediction_confidence"] > confidence_threshold
        ]

        df.loc[reduce(lambda x, y: x & y, strong_exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "strong_exit_long")
        df.loc[reduce(lambda x, y: x & y, strong_exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "strong_exit_short")

        return df

    def create_target_T(self, dataframe: pd.DataFrame) -> pd.Series:
        """
        Creates a new target (T) based on normalized future price change using ATR.
        """

        dataframe["ATR"] = ta.ATR(dataframe, timeperiod=14).bfill()  # ATR-based normalization
        dataframe["close"] = dataframe["close"].replace(0, np.nan).bfill()  # Prevent division by zero

        # ✅ Compute dynamic lookahead (ensuring valid values)
        dataframe["lookahead_dynamic"] = np.clip((dataframe["ATR"] / dataframe["close"]) * 100, 5, 20).fillna(10).astype(int)

        # ✅ Compute Future Price Change dynamically using `.apply()`
        dataframe["future_change"] = dataframe.apply(
            lambda row: dataframe["close"].shift(-int(row["lookahead_dynamic"])).iloc[row.name] - row["close"],
            axis=1
        )

        # ✅ Compute Trend Strength Using Future Price Change
        dataframe["TS"] = dataframe["future_change"].rolling(14).mean()

        # ✅ Normalize Trend Strength Using ATR + Std Dev
        dataframe["T"] = dataframe["TS"] / (
            0.5 * dataframe["ATR"] + 0.5 * dataframe["close"].rolling(14).std() + 1e-6
        )

        # ✅ Apply `tanh()` to Limit Extreme Values
        dataframe["T"] = np.tanh(dataframe["T"])

        # 🔧 Fix: No more inplace modification
        dataframe["T"] = dataframe["T"].fillna(0)

        return dataframe["T"]

    def custom_stoploss(self, pair: str, trade: 'Trade', current_time: datetime, current_rate: float,
                        current_profit: float, **kwargs) -> float:
        """
        Dynamically adjusts stoploss and ensures stoploss values exist persistently for plotting.
        """

        # ✅ Load dataframe
        dataframe, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)

        if dataframe is None or dataframe.empty:
            return self.stoploss

        last_candle = dataframe.iloc[-1]
        atr = last_candle['atr'] if 'atr' in last_candle else 0

        # ✅ Adjust ATR multiplier dynamically
        if current_profit > 0.02:
            atr_multiplier = 2.0  
        elif current_profit > 0:
            atr_multiplier = 1.5  
        elif current_profit < -0.01:
            atr_multiplier = 0.8  
        else:
            atr_multiplier = 1.2  

        buffer = atr * 0.5 if current_profit > 0.01 else 0

        # ✅ Compute stoploss
        stoploss_value = current_rate + (atr * atr_multiplier) + buffer if trade.is_short else \
                        current_rate - (atr * atr_multiplier) - buffer

        # ✅ Ensure stoploss column exists and update it directly
        if "stoploss" not in dataframe.columns:
            dataframe["stoploss"] = np.nan

        dataframe.at[last_candle.name, "stoploss"] = stoploss_value

        return stoploss_value


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
            "%-rsi-period_50_ETH/USDTUSDT_4h"
            # "%-bb_width-period_50_BTC/USDTUSDT_4h"
        }

        # Drop all columns starting with '%' unless they are in the important_features set
        columns_to_keep = [col for col in dataframe.columns if not col.startswith("%") or col in important_features]
        
        return dataframe[columns_to_keep]

