import numpy as np
import pandas as pd
import talib.abstract as ta
from sklearn.decomposition import IncrementalPCA


def create_rolling_sequences(data: pd.DataFrame, window: int = 24) -> np.ndarray:
    """
    Create rolling sequences for LSTM input.
    """
    sequences = []
    for i in range(window, len(data)):
        sequences.append(data.iloc[i - window:i].values)
    return np.array(sequences)


def add_custom_features(dataframe: pd.DataFrame) -> pd.DataFrame:
    """
    Adds specialized LSTM features: CUSUM, Hurst Exponent, Fourier Transform.
    """
    # CUSUM of price changes to detect trends
    dataframe['cusum_up'] = dataframe['close'].diff().clip(lower=0).cumsum()
    dataframe['cusum_down'] = dataframe['close'].diff().clip(upper=0).cumsum()
    
    # Hurst Exponent
    def hurst_exponent(ts, window=72):
        return np.log(np.var(ts.rolling(window).sum())) / np.log(window)
    dataframe['hurst'] = dataframe['close'].rolling(72).apply(hurst_exponent, raw=True)
    
    # Fourier Transform of price residuals
    dataframe['fourier'] = np.fft.fft(dataframe['close']).real
    
    return dataframe


def apply_dimensionality_reduction(dataframe: pd.DataFrame, n_components: int = 8) -> pd.DataFrame:
    """
    Reduces dimensionality using Incremental PCA.
    """
    ipca = IncrementalPCA(n_components=n_components)
    reduced_features = ipca.fit_transform(dataframe.dropna())
    return pd.DataFrame(reduced_features, index=dataframe.index)


def preprocess_lstm_data(dataframe: pd.DataFrame, window_size: int = 24) -> np.ndarray:
    """
    Prepares LSTM-compatible sequences.
    """
    dataframe = add_custom_features(dataframe)
    dataframe = apply_dimensionality_reduction(dataframe)
    sequences = create_rolling_sequences(dataframe, window_size)
    return sequences
