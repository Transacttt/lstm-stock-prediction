import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def load_series(csv_path: str, lookback: int = 60):
    df = pd.read_csv(csv_path, parse_dates=['Date'])
    df.sort_values('Date', inplace=True)
    close = df['Close'].values.reshape(-1, 1)

    scaler = MinMaxScaler()
    close_scaled = scaler.fit_transform(close)

    X, y = [], []
    for i in range(lookback, len(close_scaled)):
        X.append(close_scaled[i - lookback:i, 0])
        y.append(close_scaled[i, 0])
    X = np.array(X).reshape(-1, lookback, 1)
    y = np.array(y).reshape(-1, 1)
    dates = df['Date'].iloc[lookback:].reset_index(drop=True)
    return X, y, scaler, dates

