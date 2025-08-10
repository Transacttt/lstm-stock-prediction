# src/model.py
import os
try:
    # Keras 3 (preferred)
    from keras.models import Sequential, load_model
    from keras.layers import LSTM, Dense
except ImportError:
    # Fallback for older setups
    from tensorflow.keras.models import Sequential, load_model
    from tensorflow.keras.layers import LSTM, Dense

def build_lstm(input_shape):
    model = Sequential([
        LSTM(50, return_sequences=True, input_shape=input_shape),
        LSTM(50),
        Dense(1),
    ])
    model.compile(optimizer="adam", loss="mse")
    return model

def train_or_load(X, Y, ticker, epochs=3):
    os.makedirs("models", exist_ok=True)
    path = f"models/{ticker}.keras"   # modern format only
    if os.path.exists(path):
        return load_model(path, compile=False)
    model = build_lstm((X.shape[1], 1))
    model.fit(X, Y, epochs=epochs, batch_size=32, verbose=0)
    model.save(path)
    return model
