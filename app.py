import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import datetime

st.set_page_config(page_title="Anomaly Dashboard", layout="wide")
st.title("📈 Saatlik Hisse Fiyatları Anomali Tespiti Dashboard")

# Kullanıcıdan hisse kodu al
symbol = st.text_input("Hisse kodu giriniz (örn: XU100.IS)", value="XU100.IS")

# Veri çekme fonksiyonu
@st.cache_data(ttl=60)
def get_data(symbol):
    end = datetime.datetime.now()
    start = end - datetime.timedelta(days=30)  # yaklaşık 30 günlük saatlik veri
    df = yf.download(symbol, start=start, end=end, interval='1h')
    df.dropna(inplace=True)
    df['Price Change'] = df['Close'].diff()
    df.dropna(inplace=True)
    return df

# Regresyon kanalı hesaplama fonksiyonu
def regression_channel(df):
    df = df.copy()
    df['Index'] = np.arange(len(df))
    X = df[['Index']]
    y = df['Close']
    model = LinearRegression().fit(X, y)
    df['Trend'] = model.predict(X)
    df['Upper'] = df['Trend'] + (df['Close'] - df['Trend']).std()
    df['Lower'] = df['Trend'] - (df['Close'] - df['Trend']).std()
    return df

# Anomali tespiti
def detect_anomalies(df):
    model = IsolationForest(contamination=0.02, random_state=42)
    df = df.copy()
    df['Anomaly'] = model.fit_predict(df[['Price Change']])
    df['Anomaly'] = df['Anomaly'] == -1
    return df

# Ana uygulama
if symbol:
    df = get_data(symbol)
    df = regression_channel(df)
    df = detect_anomalies(df)

    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(df.index, df['Close'], label='Fiyat', color='blue')
    ax.plot(df.index, df['Trend'], label='Trend', color='black', linestyle='--')
    ax.plot(df.index, df['Upper'], color='green', linestyle=':', alpha=0.6)
    ax.plot(df.index, df['Lower'], color='red', linestyle=':', alpha=0.6)

    # Anomali noktaları
    anomalies = df[df['Anomaly']]
    ax.scatter(anomalies.index, anomalies['Close'], color='orange', alpha=0.4, label='Anomali')

    ax.set_title(f"{symbol} - Linear Regression Kanalı ve Anomaliler")
    ax.set_xlabel("Tarih")
    ax.set_ylabel("Fiyat")
    ax.legend()
    ax.grid(True)

    st.pyplot(fig)
    st.caption("Her 60 saniyede bir otomatik güncellenir.")

    # Son 5 anomaliyi tablo olarak göster
    st.subheader("🚨 Son Anomali Noktaları")
    st.dataframe(anomalies.tail(5)[['Close', 'Price Change']])






