import streamlit as st
import yfinance as yf
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.ensemble import IsolationForest
from sklearn.linear_model import LinearRegression

st.set_page_config(layout="wide")
st.title("📈 Hisse Sinyal Takibi: EMA, RSI, Anomali & Regresyon")

symbol = st.text_input("Hisse kodu giriniz (örn: XU100.IS)", value="XU100.IS")

@st.cache_data(ttl=60)
def fetch_data(ticker):
    df = yf.download(ticker, period="1d", interval="1m", progress=False, auto_adjust=True)
    if df.empty:
        return None
    df = df[['Close', 'Volume']].dropna()
    df.index = pd.to_datetime(df.index)

    # EMA ve RSI hesaplamaları
    df['EMA21'] = df['Close'].ewm(span=21, adjust=False).mean()

    delta = df['Close'].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(com=13, min_periods=14).mean()
    avg_loss = loss.ewm(com=13, min_periods=14).mean()
    rs = avg_gain / avg_loss
    df['RSI'] = 100 - (100 / (1 + rs))
    df['RSI_EMA9'] = df['RSI'].ewm(span=9, adjust=False).mean()

    # NaN satırları temizle
    df = df.dropna(subset=['Close', 'EMA21', 'RSI', 'RSI_EMA9'])

    # Anomali algılama
    iso = IsolationForest(contamination=0.10, random_state=42)
    df['anomaly'] = iso.fit_predict(df[['Close']])

    return df

df = fetch_data(symbol)

if df is None or df.empty:
    st.warning("Yeterli veri yok veya EMA/RSI hesaplanamıyor. Lütfen farklı bir hisse deneyin.")
    st.stop()

# Güvenli değer çekimi
try:
    close = df['Close'].iloc[-1].item()
    ema21 = df['EMA21'].iloc[-1].item()
    rsi = df['RSI'].iloc[-1].item()
    rsi_ema9 = df['RSI_EMA9'].iloc[-1].item()
except Exception as e:
    st.error(f"Veri okunurken hata oluştu: {e}")
    st.stop()

# Sinyal üretimi
ema_signal = "AL" if close < ema21 else "SAT"
rsi_signal = "AL" if rsi < rsi_ema9 else "SAT"

# Grafik çizimi
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

# Fiyat grafiği
ax1.plot(df.index, df['Close'], label="Close", linewidth=1)
ax1.plot(df.index, df['EMA21'], label="EMA21", linestyle="--", linewidth=1.5)
anomalies = df[df['anomaly'] == -1]
ax1.scatter(anomalies.index, anomalies['Close'], color='orange', alpha=0.3, label='Anomaly')

# Regresyon çizgisi
try:
    df['index_num'] = (df.index - df.index[0]).total_seconds().astype(float)
    X = df['index_num'].values.reshape(-1, 1)
    y = df['Close'].values.reshape(-1, 1)
    model = LinearRegression().fit(X, y)
    trend = model.predict(X).flatten()
    residuals = y.flatten() - trend
    std = residuals.std()
    upper = trend + 1.5 * std
    lower = trend - 1.5 * std
    ax1.plot(df.index, trend, color='blue', linestyle='-', label='Trend Line')
    ax1.plot(df.index, upper, color='blue', linestyle='--', label='+1.5σ')
    ax1.plot(df.index, lower, color='blue', linestyle='--', label='-1.5σ')
except Exception as e:
    st.error(f"Regresyon hatası: {e}")

ax1.set_ylabel("Fiyat")
ax1.set_title(f"{symbol} - EMA21, Anomaliler ve Regresyon")
ax1.legend()
ax1.grid(True)
ax1.text(0.99, 0.95, f"EMA Sinyali: {ema_signal}", transform=ax1.transAxes,
         fontsize=12, ha='right', va='top',
         bbox=dict(facecolor='green' if ema_signal == 'AL' else 'red', alpha=0.5))

# RSI grafiği
ax2.plot(df.index, df['RSI'], label="RSI(14)", color="purple")
ax2.plot(df.index, df['RSI_EMA9'], label="RSI EMA9", color="orange", linestyle="--")
ax2.axhline(70, color="red", linestyle="--")
ax2.axhline(30, color="green", linestyle="--")
ax2.set_ylim(0, 100)
ax2.set_ylabel("RSI")
ax2.set_title("RSI ve EMA9")
ax2.legend()
ax2.grid(True)
ax2.text(0.99, 0.95, f"RSI Sinyali: {rsi_signal}", transform=ax2.transAxes,
         fontsize=12, ha='right', va='top',
         bbox=dict(facecolor='green' if rsi_signal == 'AL' else 'red', alpha=0.5))

st.pyplot(fig)





