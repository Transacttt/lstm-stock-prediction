import streamlit as st
import pandas as pd
from src.data import load_series
from src.model import train_or_load
from src.evaluate import metrics

TICKERS = {"Apple (AAPL)": "sample_data/Apple Stock Price History.csv"}

st.set_page_config(page_title="LSTM Stock Forecaster", layout="wide")
st.title("📈 LSTM Stock Price Predictor")
st.caption("First load may take ~20 s while the model warms up.")

ticker_name = st.selectbox("Choose a stock", list(TICKERS.keys()))
csv_path    = TICKERS[ticker_name]

if st.button("Predict"):
    X, y, scaler, dates = load_series(csv_path)
    model = train_or_load(X, y, ticker_name.split()[0])

    preds_scaled = model.predict(X)
    preds = scaler.inverse_transform(preds_scaled)
    actual = scaler.inverse_transform(y)

    rmse, mae = metrics(actual, preds)
    st.metric("RMSE", f"{rmse:.2f}")
    st.metric("MAE",  f"{mae:.2f}")

    df_plot = pd.DataFrame({
        "Date": dates,
        "Actual": actual.flatten(),
        "Predicted": preds.flatten()
    })
    st.line_chart(df_plot.set_index("Date"))
