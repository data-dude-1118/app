import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Anomali Dashboard", layout="wide")
st.title("📊 Saatlik Fiyatlar | Regresyon Kanalı + Anomali Tespiti")

symbol = st.text_input("Hisse kodu giriniz (örn: XU100.IS)", value="XU100.IS")

@st.cache_data(ttl=60)
def get_data(symbol):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=30)
    df = yf.download(symbol, start=start, end=end, interval='1h')
    df.dropna(inplace=True)
    df['Price Change'] = df['Close'].diff()
    df.dropna(inplace=True)
    return df

def regression_channel(df):
    df = df.copy()
    df['Index'] = np.arange(len(df))
    X = df[['Index']]
    y = df['Close']
    model = LinearRegression().fit(X, y)
    df['Trend'] = model.predict(X)

    # Std sapma bandı
    std = (df['Close'] - df['Trend']).std()
    df['Upper'] = df['Trend'] + 1.5 * std
    df['Lower'] = df['Trend'] - 1.5 * std
    return df

def detect_anomalies(df):
    model = IsolationForest(contamination=0.15, random_state=42)
    df = df.copy()
    df['Anomaly'] = model.fit_predict(df[['Price Change']])
    df['Anomaly'] = df['Anomaly'] == -1
    return df

if symbol:
    df = get_data(symbol)
    df = regression_channel(df)
    df = detect_anomalies(df)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Fiyat', color='blue')
    ax.plot(df.index, df['Trend'], label='Trend', color='black', linestyle='--')
    ax.plot(df.index, df['Upper'], color='green', linestyle='--', label='Üst Band')
    ax.plot(df.index, df['Lower'], color='red', linestyle='--', label='Alt Band')

    anomalies = df[df['Anomaly']]
    ax.scatter(anomalies.index, anomalies['Close'], color='orange', alpha=0.4, label='Anomali')

    ax.set_title(f"{symbol} - Linear Regression Kanalı ve Anomaliler")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    st.caption("Bu grafik her 60 saniyede bir otomatik güncellenir.")

    st.subheader("🟠 Anomali Noktaları (Son 10)")
    st.dataframe(anomalies.tail(10)[['Close', 'Price Change']])






