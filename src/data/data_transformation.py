import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import numpy as np


# --- split data into training and validation split ---

def split_data(
    dataframe: pd.DataFrame,
    train_size: float
) -> tuple[np.ndarray, np.ndarray]:
    """Split time-series data into train and validation arrays."""
    if not 0 < train_size < 1:
        raise ValueError("train_size must be between 0 and 1.")

    values = dataframe.iloc[:, 0:1].values
    train_length = int(len(values) * train_size)

    df_train = values[:train_length]
    df_val = values[train_length:]
    return df_train, df_val



# --- Transform data using MinMaxScaler ---

def transform(
    df_train: np.ndarray,
    df_val: np.ndarray,
    scaler: MinMaxScaler | None = None
) -> tuple[np.ndarray, np.ndarray]:
    """Fit scaler on train, then transform train and validation."""
    scaler = scaler or MinMaxScaler()

    scaler.fit(df_train)
    df_train_scaled = scaler.transform(df_train)
    df_val_scaled = scaler.transform(df_val)
    return df_train_scaled, df_val_scaled

    


