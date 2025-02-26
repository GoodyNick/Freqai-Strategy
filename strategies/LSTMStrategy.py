import numpy as np
import pandas as pd
import torch
import talib.abstract as ta
from technical import qtpylib
from functools import reduce
from freqtrade.strategy import IStrategy
from freqtrade.persistence import Trade
from freqaimodels.lstm_model import CryptoLSTM


class LSTMStrategy(IStrategy):
    """
    FreqAI strategy using LSTM model for trade execution.
    """
    minimal_roi = {"0": 1}  # Let the model decide exits
    stoploss = -1  # Fully model-based exits
    timeframe = "1h"
    can_short = True
    use_exit_signal = True
    use_custom_stoploss = True
    process_only_new_candles = True
    startup_candle_count = 50

    def __init__(self, config):
        super().__init__(config)
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        self.model = CryptoLSTM(input_size=10, hidden_size=32, num_layers=2).to(self.device)
        self.model.load_state_dict(torch.load("lstm_model.pth", map_location=self.device))
        self.model.eval()

    def feature_engineering_expand_all(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Expands all features for FreqAI and applies filtering before training.
        """
        dataframe["%-cci-period"] = ta.CCI(dataframe, timeperiod=20)
        dataframe["%-rsi-period"] = ta.RSI(dataframe, timeperiod=10)
        dataframe["%-momentum-period"] = ta.MOM(dataframe, timeperiod=4)
        dataframe['%-ma-period'] = ta.SMA(dataframe, timeperiod=10)
        dataframe['%-macd-period'], dataframe['%-macdsignal-period'], dataframe['%-macdhist-period'] = ta.MACD(
            dataframe['close'], slowperiod=12, fastperiod=26
        )
        dataframe['%-roc-period'] = ta.ROC(dataframe, timeperiod=2)

        return dataframe

    def set_freqai_targets(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        """
        Assigns the prediction target column for FreqAI.
        """
        dataframe['&-s_target'] = self.create_target_T(dataframe)
        return dataframe

    def create_target_T(self, dataframe: pd.DataFrame):
        """
        Creates a target label (T) using scaled log future returns.
        """
        lookahead = 10
        dataframe['future_return'] = np.log(dataframe['close'].shift(-lookahead) / dataframe['close'])
        dataframe['T'] = dataframe['future_return'].rolling(window=5).mean() * 100
        dataframe['T'] = dataframe['T'].fillna(0)
        return dataframe['T']

    def populate_indicators(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        dataframe['atr'] = ta.ATR(dataframe, timeperiod=14)
        dataframe['dynamic_stoploss'] = np.clip(-dataframe['atr'] * 1.75, -0.03, -0.10)
        dataframe = self.freqai.start(dataframe, metadata, self)
        return dataframe

    def populate_entry_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        confidence_threshold = 0.8
        prediction_mean = dataframe["&-s_target"].rolling(100).mean()
        prediction_std = dataframe["&-s_target"].rolling(100).std()
        long_threshold = prediction_mean + 0.5 * prediction_std
        short_threshold = prediction_mean - 0.5 * prediction_std

        enter_long_conditions = [
            dataframe["do_predict"] == 1,
            qtpylib.crossed_above(dataframe["&-s_target"], long_threshold),
            dataframe["prediction_confidence"] > confidence_threshold,
            dataframe["volume"] > 0
        ]

        enter_short_conditions = [
            dataframe["do_predict"] == 1,
            qtpylib.crossed_below(dataframe["&-s_target"], short_threshold),
            dataframe["prediction_confidence"] > confidence_threshold,
            dataframe["volume"] > 0
        ]

        dataframe.loc[reduce(lambda x, y: x & y, enter_long_conditions), ["enter_long", "enter_tag"]] = (1, "long")
        dataframe.loc[reduce(lambda x, y: x & y, enter_short_conditions), ["enter_short", "enter_tag"]] = (1, "short")
        return dataframe

    def populate_exit_trend(self, dataframe: pd.DataFrame, metadata: dict) -> pd.DataFrame:
        confidence_threshold = 0.85
        exit_threshold = dataframe["&-s_target"].rolling(100).median()

        strong_exit_long_conditions = [
            dataframe["do_predict"] == 1,
            qtpylib.crossed_below(dataframe["&-s_target"], exit_threshold),
            dataframe["prediction_confidence"] > confidence_threshold
        ]

        strong_exit_short_conditions = [
            dataframe["do_predict"] == 1,
            qtpylib.crossed_above(dataframe["&-s_target"], exit_threshold),
            dataframe["prediction_confidence"] > confidence_threshold
        ]

        dataframe.loc[reduce(lambda x, y: x & y, strong_exit_long_conditions), ["exit_long", "exit_tag"]] = (1, "strong_exit_long")
        dataframe.loc[reduce(lambda x, y: x & y, strong_exit_short_conditions), ["exit_short", "exit_tag"]] = (1, "strong_exit_short")
        return dataframe
