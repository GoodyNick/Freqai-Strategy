import logging
from functools import reduce
from typing import Dict
import joblib
import os

import numpy as np
import pandas as pd
import talib.abstract as ta
from pandas import DataFrame
from technical import qtpylib
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import Ridge

from freqtrade.exchange.exchange_utils import *
from freqtrade.strategy import IStrategy, RealParameter

logger = logging.getLogger(__name__)


class ExampleLSTMStrategy_v1(IStrategy):
    """
    This is an example strategy that uses the LSTMRegressor model to predict the target score.
    Use at your own risk.
    This is a simple example strategy and should be used for educational purposes only.
    """
    plot_config = {
        "main_plot": {},
        "subplots": {
            "predictions": {  
                "&-target_mean": {"color": "blue", "plot_type": "line"},  # Model's expected target
                "&-target": {"color": "brown", "plot_type": "line"},  # Actual target values
                "do_predict": {"color": "black", "plot_type": "scatter"},  # Show predictions as dots
            },
        },
    }
    
    # Hyperspace parameters:
    buy_params = {
        "threshold_buy": 0.59453,
        "w0": 0.54347,
        "w1": 0.82226,
        "w2": 0.56675,
        "w3": 0.77918,
        "w4": 0.98488,
        "w5": 0.31368,
        "w6": 0.75916,
        "w7": 0.09226,
        "w8": 0.85667,
    }
    sell_params = {
        "threshold_sell": 0.80573,
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

    # Weights for calculating the aggregate score - the sum of all weighted normalized indicators has to be 1!
    w0 = RealParameter(0, 1, default=0.10, space='buy')
    w1 = RealParameter(0, 1, default=0.15, space='buy')
    w2 = RealParameter(0, 1, default=0.10, space='buy')
    w3 = RealParameter(0, 1, default=0.15, space='buy')
    w4 = RealParameter(0, 1, default=0.10, space='buy')
    w5 = RealParameter(0, 1, default=0.10, space='buy')
    w6 = RealParameter(0, 1, default=0.10, space='buy')
    w7 = RealParameter(0, 1, default=0.05, space='buy')
    w8 = RealParameter(0, 1, default=0.15, space='buy')

    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    process_only_new_candles = True

    startup_candle_count = 20

    do_normalize = True # Set to True to normalize the indicators
    dynamic_target_weights = False  # Set to True to enable dynamic calculation of target weights
    prediction_metrics_storage = []  # Class-level storage for all pairs

    def feature_engineering_expand_all(self, dataframe: DataFrame, period: int,
                                       metadata: Dict, **kwargs):

        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12,
            fastperiod=26)
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        bollinger = qtpylib.bollinger_bands(
            qtpylib.typical_price(dataframe), window=period, stds=2.2
        )
        dataframe["bb_lowerband-period"] = bollinger["lower"]
        dataframe["bb_middleband-period"] = bollinger["mid"]
        dataframe["bb_upperband-period"] = bollinger["upper"]
        dataframe["%-bb_width-period"] = (
                                                 dataframe["bb_upperband-period"]
                                                 - dataframe["bb_lowerband-period"]
                                         ) / dataframe["bb_middleband-period"]
        dataframe["%-close-bb_lower-period"] = (
                dataframe["close"] / dataframe["bb_lowerband-period"]
        )

        return dataframe

    def feature_engineering_expand_basic(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe["%-pct-change"] = dataframe["close"].pct_change()
        dataframe["%-raw_volume"] = dataframe["volume"]
        dataframe["%-raw_price"] = dataframe["close"]

        if self.do_normalize:
            numeric_features = [col for col in dataframe.columns if col.startswith("%-") and dataframe[col].dtype in [np.float64, np.int64]]
            logger.info(f"🔍 Normalizing {len(numeric_features)} features: {numeric_features}")
            if numeric_features:
                scaler = StandardScaler()
                dataframe[numeric_features] = scaler.fit_transform(dataframe[numeric_features])
                # ✅ Save scaler for consistent feature scaling in live trading
                joblib.dump({"scaler": scaler, "features": numeric_features}, "./user_data/freqai_scaler.pkl")
                logger.info(f"✅ StandardScaler applied & saved for {len(numeric_features)} features.")

        return dataframe

    def feature_engineering_standard(self, dataframe: DataFrame, metadata: Dict, **kwargs):

        dataframe['date'] = pd.to_datetime(dataframe['date'])
        dataframe["%-day_of_week"] = dataframe["date"].dt.dayofweek
        dataframe["%-hour_of_day"] = dataframe["date"].dt.hour
        return dataframe

    def set_freqai_targets(self, dataframe: DataFrame, metadata: Dict, **kwargs) -> DataFrame:

        dataframe['ma'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['roc'] = ta.ROC(dataframe, timeperiod=2)
        dataframe['macd'], dataframe['macdsignal'], dataframe['macdhist'] = ta.MACD(
            dataframe['close'], slowperiod=12, fastperiod=26)
        dataframe['momentum'] = ta.MOM(dataframe, timeperiod=4)
        dataframe['rsi'] = ta.RSI(dataframe, timeperiod=10)
        bollinger = ta.BBANDS(dataframe, timeperiod=20)
        dataframe['bb_upperband'] = bollinger['upperband']
        dataframe['bb_middleband'] = bollinger['middleband']
        dataframe['bb_lowerband'] = bollinger['lowerband']
        dataframe['cci'] = ta.CCI(dataframe, timeperiod=20)
        dataframe['stoch'] = ta.STOCH(dataframe)['slowk']
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['obv'] = ta.OBV(dataframe)

        if self.dynamic_target_weights:    
            # ✅ Step 1: Select Indicators for Target Calculation (No Prefix)
            target_input_columns = [
                "ma", "roc", "macd", "momentum", "rsi",
                "cci", "stoch", "atr", "obv"
            ]

            # ✅ Step 2: Normalize These Features for Aggregation
            for feature in target_input_columns:
                dataframe[f"normalized_{feature}"] = (
                    dataframe[feature] - dataframe[feature].rolling(14).mean()
                ) / dataframe[feature].rolling(14).std()

            # ✅ Step 3: Train Ridge Regression to Learn Best Weights
            X = dataframe[[f"normalized_{feature}" for feature in target_input_columns]].fillna(0)
            y = dataframe["close"].pct_change().fillna(0)  # Target proxy: price change

            model = Ridge(alpha=1.0)
            model.fit(X, y)

            # Extract & Normalize Weights
            feature_weights = dict(zip(target_input_columns, model.coef_))
            total_weight = sum(abs(v) for v in feature_weights.values())
            normalized_weights = {k: v / total_weight for k, v in feature_weights.items()}

            # ✅ Step 4: Aggregate Features Using Learned Weights
            dataframe["S"] = sum(
                dataframe[f"normalized_{feature}"] * weight for feature, weight in normalized_weights.items()
            )

            # ✅ Step 5: Market Regime Filters (No Prefix Needed)
            dataframe['R'] = 0
            dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
            dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

            dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
            dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)

            # ✅ Step 6: Volatility Adjustment
            bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
            dataframe['V'] = 1 / bb_width
            dataframe['V2'] = 1 / dataframe['atr']

            # ✅ Step 7: Compute Final Target Value (`&-target`)
            dataframe['&-target'] = dataframe["S"] * dataframe['R'] * dataframe['V'] * dataframe['R2'] * dataframe['V2']

            logger.info(f"🔍 Learned Target Weights: {normalized_weights}")
        else:
            # Step 1: Normalize Indicators:
            # Why? Normalizing the indicators will make them comparable and allow us to assign weights to them.
            # How? We will calculate the z-score of each indicator by subtracting the rolling mean and dividing by the
            # rolling standard deviation. This will give us a normalized value that is centered around 0 with a standard
            # deviation of 1.
            dataframe['normalized_stoch'] = (dataframe['stoch'] - dataframe['stoch'].rolling(window=14).mean()) / dataframe[
                'stoch'].rolling(window=14).std()
            dataframe['normalized_atr'] = (dataframe['atr'] - dataframe['atr'].rolling(window=14).mean()) / dataframe[
                'atr'].rolling(window=14).std()
            dataframe['normalized_obv'] = (dataframe['obv'] - dataframe['obv'].rolling(window=14).mean()) / dataframe[
                'obv'].rolling(window=14).std()
            dataframe['normalized_ma'] = (dataframe['close'] - dataframe['close'].rolling(window=10).mean()) / dataframe[
                'close'].rolling(window=10).std()
            dataframe['normalized_macd'] = (dataframe['macd'] - dataframe['macd'].rolling(window=26).mean()) / dataframe[
                'macd'].rolling(window=26).std()
            dataframe['normalized_roc'] = (dataframe['roc'] - dataframe['roc'].rolling(window=2).mean()) / dataframe[
                'roc'].rolling(window=2).std()
            dataframe['normalized_momentum'] = (dataframe['momentum'] - dataframe['momentum'].rolling(window=4).mean()) / \
                                            dataframe['momentum'].rolling(window=4).std()
            dataframe['normalized_rsi'] = (dataframe['rsi'] - dataframe['rsi'].rolling(window=10).mean()) / dataframe[
                'rsi'].rolling(window=10).std()
            dataframe['normalized_bb_width'] = (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(
                window=20).mean() / (dataframe['bb_upperband'] - dataframe['bb_lowerband']).rolling(window=20).std()
            dataframe['normalized_cci'] = (dataframe['cci'] - dataframe['cci'].rolling(window=20).mean()) / dataframe[
                'cci'].rolling(window=20).std()

            # Dynamic Weights (Example: Increase the weight of momentum in a strong trend)
            trend_strength = abs(dataframe['ma'] - dataframe['close'])

            # Calculate the rolling mean and standard deviation of the trend strength to determine a strong trend
            # The threshold is set to 1.5 times the standard deviation above the mean, but can be adjusted as needed
            strong_trend_threshold = trend_strength.rolling(window=14).mean() + 1.5 * trend_strength.rolling(
                window=14).std()

            # Assign a higher weight to momentum if the trend is strong
            is_strong_trend = trend_strength > strong_trend_threshold

            # Assign the dynamic weights to the dataframe
            dataframe['w_momentum'] = np.where(is_strong_trend, self.w3.value * 1.5, self.w3.value)

            # Step 2: Calculate aggregate score S
            w = [self.w0.value, self.w1.value, self.w2.value, self.w3.value, self.w4.value, self.w5.value,
                self.w6.value, self.w7.value, self.w8.value]

            dataframe['S'] = w[0] * dataframe['normalized_ma'] + w[1] * dataframe['normalized_macd'] + w[2] * dataframe[
                'normalized_roc'] + w[3] * dataframe['normalized_rsi'] + w[4] * \
                            dataframe['normalized_bb_width'] + w[5] * dataframe['normalized_cci'] + dataframe[
                                'w_momentum'] * dataframe['normalized_momentum'] + self.w8.value * dataframe[
                                'normalized_stoch'] + self.w7.value * dataframe['normalized_atr'] + self.w6.value * \
                            dataframe['normalized_obv']

            # Step 3: Market Regime Filter R
            # EXPLANATION: If the price is above the upper Bollinger Band, assign a value
            # of 1 to R. If the price is below the lower Bollinger Band, assign a value of -1 to R. Otherwise,
            # the value R stays 0.
            # What's basically happening here is that we are assigning a value of 1 to R when
            # the price is in the upper band, -1 when the price is in the lower band, and 0 when the price is in the
            # middle band. This is a simple way to determine the market regime based on Bollinger Bands. What is market
            # regime? Market regime is the state of the market. It can be trending, ranging, or reversing. So we are
            # using Bollinger Bands to determine the market regime. You can use other indicators to determine the market
            # regime as well. For example, you can use moving averages, RSI, MACD, etc.
            dataframe['R'] = 0
            dataframe.loc[(dataframe['close'] > dataframe['bb_middleband']) & (
                    dataframe['close'] > dataframe['bb_upperband']), 'R'] = 1
            dataframe.loc[(dataframe['close'] < dataframe['bb_middleband']) & (
                    dataframe['close'] < dataframe['bb_lowerband']), 'R'] = -1

            # Additional Market Regime Filter based on long-term MA
            dataframe['ma_100'] = ta.SMA(dataframe, timeperiod=100)
            dataframe['R2'] = np.where(dataframe['close'] > dataframe['ma_100'], 1, -1)

            # Step 4: Volatility Adjustment V
            # EXPLANATION: Calculate the Bollinger Band width and assign it to V. The Bollinger Band width is the
            # difference between the upper and lower Bollinger Bands divided by the middle Bollinger Band. The idea is
            # that when the Bollinger Bands are wide, the market is volatile, and when the Bollinger Bands are narrow,
            # the market is less volatile. So we are using the Bollinger Band width as a measure of volatility. You can
            # use other indicators to measure volatility as well. For example, you can use the ATR (Average True Range)
            bb_width = (dataframe['bb_upperband'] - dataframe['bb_lowerband']) / dataframe['bb_middleband']
            dataframe['V'] = 1 / bb_width  # example, assuming V is inversely proportional to BB width

            # Another Volatility Adjustment using ATR
            dataframe['V2'] = 1 / dataframe['atr']

            # Get Final Target Score to incorporate new calculations
            dataframe['T'] = dataframe['S'] * dataframe['R'] * dataframe['V'] * dataframe['R2'] * dataframe['V2']

            # Assign the target score T to the AI target column
            dataframe['&-target'] = dataframe['T']

        if self.do_normalize:
            # ✅ Normalize Targets (`&-`) Using Rolling Mean & Std
            target_columns = [col for col in dataframe.columns if col.startswith("&-") and dataframe[col].dtype in [np.float64, np.int64]]
            if target_columns:
                for target in target_columns:
                    dataframe[target] = (dataframe[target] - dataframe[target].rolling(50).mean()) / dataframe[target].rolling(50).std()
                logger.info(f"✅ Rolling Mean & Std Normalization Applied for {len(target_columns)} Targets")

        return dataframe

    def populate_indicators(self, dataframe: DataFrame, metadata: dict) -> DataFrame:

        self.freqai_info = self.config["freqai"]

        if self.do_normalize:
            # ✅ Load saved scaler for feature normalization in live predictions
            try:
                saved_data = joblib.load("./user_data/freqai_scaler.pkl")
                scaler, trained_features = saved_data["scaler"], saved_data["features"]

                for f in trained_features:
                    if f not in dataframe.columns:
                        dataframe[f] = 0  # Fill missing features with zero

                dataframe[trained_features] = scaler.transform(dataframe[trained_features])
                logger.info("✅ Applied StandardScaler to live feature data.")

            except FileNotFoundError:
                logger.warning("⚠️ Scaler file not found! Model may behave inconsistently.")

        dataframe = self.freqai.start(dataframe, metadata, self)

        # ✅ Adjust DI_threshold after FreqAI processes the data
        if self.freqai_info["feature_parameters"]["DI_threshold"] == 1.0:  # If using default value
            asset_volatility = dataframe["close"].pct_change().rolling(50).std().mean()
            di_base = 3.0  # Default DI_threshold when volatility is low
            di_threshold = di_base / (1 + asset_volatility * 5)  # Inverse scaling
            self.freqai_info["feature_parameters"]["DI_threshold"] = di_threshold  
            logger.info(f"🔍 Applied Dynamic DI_threshold for {metadata['pair']}: {di_threshold:.2f}")
        logger.info(f"🔍 [DEBUG] Current DI_threshold for {metadata['pair']}: {self.freqai_info['feature_parameters']['DI_threshold']}")

        self.compute_prediction_metrics(dataframe, metadata)
        self.save_prediction_metrics()
        
        return dataframe

    def populate_entry_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_factor = df["prediction_confidence"]

        enter_long_conditions = [
            df["do_predict"] == 1,
            df["&-target"] > (df["&-target_mean"] * confidence_factor),  # Dynamic threshold
            df["volume"] > 0
        ]

        enter_short_conditions = [
            df["do_predict"] == 1,
            df["&-target"] < (df["&-target_mean"] * confidence_factor),
            df["volume"] > 0
        ]

        df.loc[
            reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]
        ] = (1, "long")

        df.loc[
            reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]
        ] = (1, "short")

        return df

    def populate_exit_trend(self, df: DataFrame, metadata: dict) -> DataFrame:
        confidence_threshold = 0.85  # Adjust dynamically

        exit_long_conditions = [
            df["do_predict"] == 1,
            df["&-target"] < df["&-target_mean"],  # Model predicts reversal
            df["prediction_confidence"] > confidence_threshold,  # High confidence in reversal
        ]

        exit_short_conditions = [
            df["do_predict"] == 1,
            df["&-target"] > df["&-target_mean"],
            df["prediction_confidence"] > confidence_threshold,
        ]

        if exit_long_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_long_conditions), ["exit_long", "exit_tag"]
            ] = (1, "exit_long")

        if exit_short_conditions:
            df.loc[
                reduce(lambda x, y: x & y, exit_short_conditions), ["exit_short", "exit_tag"]
            ] = (1, "exit_short")

        return df
    
    def confirm_trade_entry(self, pair: str, order_type: str, amount: float, rate: float, time_in_force: str, 
                        current_time, entry_tag, side: str, **kwargs) -> bool:

        df, _ = self.dp.get_analyzed_dataframe(pair, self.timeframe)
        last_candle = df.iloc[-1]

        confidence = last_candle["prediction_confidence"] if "prediction_confidence" in last_candle else 0.5

        # Adjust position size dynamically (scale trade size with confidence)
        adjusted_size = amount * confidence

        return super().confirm_trade_entry(pair, order_type, adjusted_size, rate, time_in_force, 
                                        current_time, entry_tag, side, **kwargs)

    def compute_prediction_metrics(self, dataframe: pd.DataFrame, metadata: dict, target_col: str = "&-target"): 
        """
        Computes and stores prediction accuracy metrics for all trading pairs.
        Saves the results to a CSV file after backtesting.
        """
        prediction_col = target_col + "_mean"

        # Ensure required columns exist
        if prediction_col not in dataframe.columns:
            logger.warning(f"❌ Column '{prediction_col}' not found in dataframe. Skipping prediction metrics.")
            return dataframe

        # ✅ Step 1: Directional Accuracy (Sign Match)
        dataframe["prediction_correct"] = (np.sign(dataframe[target_col]) == np.sign(dataframe[prediction_col])).astype(int)

        # ✅ Step 2: Rolling Accuracy (Last 50 candles)
        dataframe["rolling_accuracy"] = dataframe["prediction_correct"].rolling(50, min_periods=1).mean()

        # ✅ Step 3: Mean Absolute Error (MAE)
        dataframe["mae"] = np.abs(dataframe[target_col] - dataframe[prediction_col]).rolling(100, min_periods=1).mean()

        # ✅ Step 4: Prediction Confidence (Normalized by Standard Deviation)
        std_col = target_col + "_std"
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
        total_predictions = dataframe["do_predict"].sum()
        total_targets_available = dataframe[target_col].notna().sum()
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
            "correlation": dataframe[target_col].corr(dataframe[prediction_col])  # ✅ Step 8: Correlation between Target and Predictions
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
