import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
import matplotlib.pyplot as plt
import datetime

st.set_page_config(layout="wide")
st.title("📈 Hisse Anomali & Regresyon Kanal Takibi")
symbol = st.text_input("Hisse kodu giriniz (örn: XU100.IS):", "XU100.IS")

@st.cache_data(ttl=60)
def fetch_minute_data(symbol):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=6)  # Son 6 saatlik veri
    data = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
    data.dropna(inplace=True)
    return data

def compute_regression_channel(df):
    df = df.copy()
    df["Timestamp"] = np.arange(len(df))
    X = df["Timestamp"].values.reshape(-1, 1)
    y = df["Close"].values
    coeffs = np.polyfit(df["Timestamp"], df["Close"], deg=1)
    y_pred = coeffs[0] * df["Timestamp"] + coeffs[1]
    residuals = y - y_pred
    std = np.std(residuals)

    df["RegLine"] = y_pred
    df["Upper"] = y_pred + 2 * std
    df["Lower"] = y_pred - 2 * std
    return df

def detect_anomalies(df):
    df = df.copy()
    df["Returns"] = df["Close"].diff()
    df.dropna(inplace=True)
    X = df[["Returns"]].values

    model = IsolationForest(contamination=0.15, random_state=42)
    df["Anomaly"] = model.fit_predict(X)
    df["Anomaly"] = df["Anomaly"].apply(lambda x: x == -1)
    return df

def plot_chart(df):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df.index, df["Close"], label="Fiyat", linewidth=1)
    ax.plot(df.index, df["RegLine"], label="Regresyon", linestyle="--")
    ax.fill_between(df.index, df["Upper"], df["Lower"], color="gray", alpha=0.1, label="Regresyon Kanalı")
    
    anomalies = df[df["Anomaly"]]
    ax.scatter(anomalies.index, anomalies["Close"], color="orange", label="Anomali", alpha=0.5)

    ax.set_title(f"{symbol} - Dakikalık Grafik")
    ax.set_xlabel("Zaman")
    ax.set_ylabel("Fiyat")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

# Uygulama çalışıyor:
try:
    df = fetch_minute_data(symbol)
    df = compute_regression_channel(df)
    df = detect_anomalies(df)
    plot_chart(df)
    st.caption("Veri her 1 dakikada bir yenilenir.")
except Exception as e:
    st.error(f"Bir hata oluştu: {e}")








