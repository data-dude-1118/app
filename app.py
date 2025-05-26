import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import IsolationForest
import datetime

st.set_page_config(layout="wide")
st.title("📈 Hisse Fiyatı: Regresyon Kanalı & Anomali Tespiti")

symbol = st.text_input("Hisse kodu giriniz (örn: XU100.IS):", "XU100.IS")

@st.cache_data(ttl=60)
def fetch_minute_data(symbol):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(hours=6)
    df = yf.download(symbol, start=start, end=end, interval="1m", progress=False)
    df.dropna(inplace=True)
    return df

def add_regression_channel(df):
    df = df.copy().reset_index()
    df['Timestamp'] = df['Datetime'].astype(np.int64) // 10 ** 9  # saniyeye çevir
    X = df['Timestamp'].values.reshape(-1, 1)
    y = df['Close'].values.reshape(-1, 1)

    model = LinearRegression().fit(X, y)
    df['RegLine'] = model.predict(X)
    
    residuals = y.flatten() - df['RegLine']
    std = np.std(residuals)
    df['Upper'] = df['RegLine'] + 2 * std
    df['Lower'] = df['RegLine'] - 2 * std
    
    return df

def detect_anomalies(df):
    df = df.copy()
    df['Return'] = df['Close'].pct_change().fillna(0)
    X = df[['Return']]
    
    model = IsolationForest(contamination=0.15, random_state=42)
    df['Anomaly'] = model.fit_predict(X)
    df['Anomaly'] = df['Anomaly'].map({1: 0, -1: 1})  # 1 = anomaly
    return df

if symbol:
    df = fetch_minute_data(symbol)

    if df.empty:
        st.warning("Veri alınamadı.")
    else:
        df = add_regression_channel(df)
        df = detect_anomalies(df)

        fig, ax = plt.subplots(figsize=(14, 6))
        ax.plot(df['Datetime'], df['Close'], label="Fiyat", color='blue', linewidth=1)
        ax.plot(df['Datetime'], df['RegLine'], label="Regresyon", linestyle='--', color='black')
        ax.plot(df['Datetime'], df['Upper'], label="Üst Kanal", linestyle=':', color='green')
        ax.plot(df['Datetime'], df['Lower'], label="Alt Kanal", linestyle=':', color='red')

        anomalies = df[df['Anomaly'] == 1]
        ax.scatter(anomalies['Datetime'], anomalies['Close'], color='orange', alpha=0.5, s=50, label="Anomali")

        ax.set_title(f"{symbol} - Dakikalık Fiyatlar")
        ax.set_xlabel("Zaman")
        ax.set_ylabel("Fiyat")
        ax.legend()
        ax.grid(True)

        st.pyplot(fig)




