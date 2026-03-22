
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Gold Price Prediction", page_icon="win ", layout="wide")

st.title(" Gold Price Prediction using Machine Learning")
st.markdown("### Predicting XAU/USD using Baseline, ARIMA & LSTM models")

st.sidebar.header(" Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Select Model", ["Baseline Naive", "ARIMA", "LSTM"])

st.sidebar.markdown("---")
st.sidebar.markdown("Made with  for Predictive Analytics Course")

@st.cache_data
def load_data(start, end):
    df = yf.download("GC=F", start=start, end=end)
    df = df[["Close"]]
    df.dropna(inplace=True)
    return df

with st.spinner("Fetching gold price data..."):
    df = load_data(start_date, end_date)

st.subheader(" Gold Price Data")
st.line_chart(df["Close"])

st.subheader(" Data Summary")
st.dataframe(df.describe())

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

def evaluate(actual, predicted):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2

if model_choice == "Baseline Naive":
    st.subheader(" Baseline Naive Model")
    naive_pred = test["Close"].shift(1).dropna()
    test_b = test.iloc[1:].copy()
    mae, rmse, r2 = evaluate(test_b["Close"], naive_pred)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(test_b.index, test_b["Close"], label="Actual", color="gold")
    ax.plot(test_b.index, naive_pred, label="Predicted", color="red", linestyle="--")
    ax.legend()
    ax.set_title("Baseline Naive Model vs Actual")
    st.pyplot(fig)

elif model_choice == "ARIMA":
    st.subheader(" ARIMA Model")
    with st.spinner("Training ARIMA model..."):
        arima_model = ARIMA(train["Close"], order=(5,1,0))
        arima_result = arima_model.fit()
        arima_pred = arima_result.forecast(steps=len(test))
        arima_pred.index = test.index
    mae, rmse, r2 = evaluate(test["Close"], arima_pred)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(test.index, test["Close"], label="Actual", color="gold")
    ax.plot(test.index, arima_pred, label="ARIMA Prediction", color="blue", linestyle="--")
    ax.legend()
    ax.set_title("ARIMA Model vs Actual")
    st.pyplot(fig)

elif model_choice == "LSTM":
    st.subheader(" LSTM Model")
    with st.spinner("Training LSTM model... this takes a minute "):
        scaler = MinMaxScaler(feature_range=(0,1))
        scaled = scaler.fit_transform(df[["Close"]])

        def create_sequences(data, seq=60):
            X, y = [], []
            for i in range(seq, len(data)):
                X.append(data[i-seq:i, 0])
                y.append(data[i, 0])
            return np.array(X), np.array(y)

        X, y = create_sequences(scaled)
        split = int(len(X) * 0.8)
        X_train, X_test = X[:split], X[split:]
        y_train, y_test = y[:split], y[split:]
        X_train = X_train.reshape(-1, 60, 1)
        X_test = X_test.reshape(-1, 60, 1)

        model = Sequential([
            LSTM(50, return_sequences=True, input_shape=(60,1)),
            Dropout(0.2),
            LSTM(50),
            Dropout(0.2),
            Dense(25),
            Dense(1)
        ])
        model.compile(optimizer="adam", loss="mean_squared_error")
        model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=0)

        pred_scaled = model.predict(X_test)
        lstm_pred = scaler.inverse_transform(pred_scaled)
        y_actual = scaler.inverse_transform(y_test.reshape(-1,1))

    mae, rmse, r2 = evaluate(y_actual, lstm_pred)

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(y_actual, label="Actual", color="gold")
    ax.plot(lstm_pred, label="LSTM Prediction", color="green", linestyle="--")
    ax.legend()
    ax.set_title("LSTM Model vs Actual")
    st.pyplot(fig)

st.subheader(" Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")
