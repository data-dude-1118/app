import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression
import datetime

st.set_page_config(page_title="Hisse Sinyal Takibi", layout="wide")

st.title("📊 Hisse Sinyal Takibi: EMA, RSI, Regresyon Kanalı, Anomali")

# --- Kullanıcıdan hisse kodu al
symbol = st.text_input("Hisse kodu (örn: XU100.IS)", value="XU100.IS").upper()

# --- RSI Hesabı
def compute_rsi(series, period=14):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=period - 1, min_periods=period).mean()
    avg_loss = loss.ewm(com=period - 1, min_periods=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

# --- Sinyal Hesaplama
def compute_signals(df):
    try:
        price = df['Close'].iloc[-1]
        ema21 = df['EMA21'].iloc[-1]
        rsi = df['RSI'].iloc[-1]
        rsi_ema = df['RSI_EMA9'].iloc[-1]

        ema_signal = "AL" if price < ema21 else "SAT"
        rsi_signal = "AL" if rsi < rsi_ema else "SAT"
        return ema_signal, rsi_signal
    except:
        return "NO", "NO"

# --- Veri Çekme (cache ile 1 dk güncelleme)
@st.cache_data(ttl=60)
def fetch_data(symbol):
    df = yf.download(symbol, period="1d", interval="1m", progress=False)
    df = df[['Close']].dropna()
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()
    df['RSI'] = compute_rsi(df['Close'])
    df['RSI_EMA9'] = df['RSI'].ewm(span=9, adjust=False).mean()
    return df.dropna()

# --- Çizim
if symbol:
    try:
        df = fetch_data(symbol)

        # --- Anomaly Detection
        iso = IsolationForest(contamination=0.10, random_state=42)
        df['anomaly'] = iso.fit_predict(df[['Close']])
        anomalies = df[df['anomaly'] == -1]

        # --- Linear Regression Channel
        df['index_num'] = (df.index - df.index[0]).total_seconds()
        X = df['index_num'].values.reshape(-1, 1)
        y = df['Close'].values.reshape(-1, 1)
        model = LinearRegression().fit(X, y)
        trend = model.predict(X).flatten()
        std = (df['Close'] - trend).std()
        upper = trend + 1.5 * std
        lower = trend - 1.5 * std

        # --- Sinyal Bilgileri
        ema_signal, rsi_signal = compute_signals(df)

        # --- GRAFİK
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

        # Fiyat + EMA + Anomali + Trend
        ax1.plot(df.index, df['Close'], label='Close (1m)', color='blue', linewidth=1)
        ax1.plot(df.index, df['EMA21'], label='EMA21', linestyle='--', color='purple')
        ax1.plot(df.index, trend, label='Trend', color='black')
        ax1.plot(df.index, upper, label='+1.5σ', linestyle='--', color='green')
        ax1.plot(df.index, lower, label='-1.5σ', linestyle='--', color='red')
        ax1.scatter(anomalies.index, anomalies['Close'], color='orange', alpha=0.4, label='Anomali', zorder=5)

        ax1.set_ylabel("Fiyat")
        ax1.set_title(f"{symbol} - EMA, Anomali ve Regresyon Kanalı")
        ax1.legend()
        ax1.grid(True)

        # RSI
        ax2.plot(df.index, df['RSI'], label='RSI(14)', color='blue')
        ax2.plot(df.index, df['RSI_EMA9'], label='RSI EMA9', color='orange', linestyle='--')
        ax2.axhline(70, color='red', linestyle='--')
        ax2.axhline(30, color='green', linestyle='--')
        ax2.set_ylim(0, 100)
        ax2.set_ylabel("RSI")
        ax2.set_xlabel("Zaman")
        ax2.legend()
        ax2.grid(True)

        st.pyplot(fig)

        # --- Sinyaller
        st.markdown(f"### 🔔 EMA Sinyali: {'🟢 AL' if ema_signal == 'AL' else '🔴 SAT'}")
        st.markdown(f"### 📶 RSI Sinyali: {'🟢 AL' if rsi_signal == 'AL' else '🔴 SAT'}")

        st.caption("📡 Her 1 dakikada bir otomatik yenilenir.")
    except Exception as e:
        st.error(f"Hata oluştu: {e}")






