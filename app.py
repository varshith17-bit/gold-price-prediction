import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import yfinance as yf
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import warnings
warnings.filterwarnings("ignore")

st.set_page_config(page_title="Gold Price Prediction", page_icon="🥇", layout="wide")

st.title("🥇 Gold Price Prediction using Machine Learning")
st.markdown("### Predicting XAU/USD using Baseline & ARIMA models")

st.sidebar.header("⚙️ Settings")
start_date = st.sidebar.date_input("Start Date", pd.to_datetime("2015-01-01"))
end_date = st.sidebar.date_input("End Date", pd.to_datetime("2024-12-31"))
model_choice = st.sidebar.selectbox("Select Model", ["Baseline Naive", "ARIMA"])
st.sidebar.markdown("---")
st.sidebar.markdown("Made with ❤️ for Predictive Analytics Course")

@st.cache_data
def load_data(start, end):
    df = yf.download("GC=F", start=start, end=end)
    df = df[["Close"]]
    df.dropna(inplace=True)
    df.columns = ["Close"]
    return df

with st.spinner("Fetching gold price data..."):
    df = load_data(start_date, end_date)

st.subheader("📈 Gold Price Data")
st.line_chart(df["Close"])

st.subheader("📊 Data Summary")
st.dataframe(df.describe())

train_size = int(len(df) * 0.8)
train = df.iloc[:train_size]
test = df.iloc[train_size:]

def evaluate(actual, predicted, model_name):
    mae = mean_absolute_error(actual, predicted)
    rmse = np.sqrt(mean_squared_error(actual, predicted))
    r2 = r2_score(actual, predicted)
    return mae, rmse, r2

if model_choice == "Baseline Naive":
    st.subheader("🔴 Baseline Naive Model")
    naive_pred = test["Close"].shift(1).dropna()
    test_b = test.iloc[1:].copy()
    mae, rmse, r2 = evaluate(test_b["Close"], naive_pred, "Baseline")

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(test_b.index, test_b["Close"], label="Actual", color="gold")
    ax.plot(test_b.index, naive_pred, label="Predicted", color="red", linestyle="--")
    ax.legend()
    ax.set_title("Baseline Naive Model vs Actual")
    st.pyplot(fig)

elif model_choice == "ARIMA":
    st.subheader("🔵 ARIMA Model")
    with st.spinner("Training ARIMA model... please wait ⏳"):
        arima_model = ARIMA(train["Close"], order=(5,1,0))
        arima_result = arima_model.fit()
        arima_pred = arima_result.forecast(steps=len(test))
        arima_pred.index = test.index
    mae, rmse, r2 = evaluate(test["Close"], arima_pred, "ARIMA")

    fig, ax = plt.subplots(figsize=(12,4))
    ax.plot(test.index, test["Close"], label="Actual", color="gold")
    ax.plot(test.index, arima_pred, label="ARIMA Prediction", color="blue", linestyle="--")
    ax.legend()
    ax.set_title("ARIMA Model vs Actual")
    st.pyplot(fig)

st.subheader("📊 Model Performance Metrics")
col1, col2, col3 = st.columns(3)
col1.metric("MAE", f"{mae:.2f}")
col2.metric("RMSE", f"{rmse:.2f}")
col3.metric("R² Score", f"{r2:.4f}")
