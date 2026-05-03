import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from data_processor import DataProcessor
from model_trainer import ModelTrainer, TENSORFLOW_AVAILABLE
import os
from datetime import datetime, timedelta

# --- Page Configuration ---
st.set_page_config(
    page_title="FinSight | Indian Stock Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# --- Custom CSS for Premium Look ---
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stMetric {
        background-color: #1e2130;
        padding: 15px;
        border-radius: 10px;
        box-shadow: 0 4px 6px rgba(0,0,0,0.1);
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 20px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 50px;
        white-space: pre-wrap;
        background-color: #1e2130;
        border-radius: 5px;
        color: white;
        padding-left: 20px;
        padding-right: 20px;
    }
    .stTabs [aria-selected="true"] {
        background-color: #4e79a7;
    }
</style>
""", unsafe_allow_html=True)

# --- Header ---
st.title("📊 FinSight: Indian Stock Market Analytics")
st.markdown("---")

# --- Sidebar ---
st.sidebar.image("https://cdn-icons-png.flaticon.com/512/2681/2681826.png", width=100)
st.sidebar.title("Dashboard Controls")

ticker_list = ["RELIANCE", "TCS", "HDFCBANK", "INFY", "ICICIBANK"]
ticker = st.sidebar.selectbox("Select Asset", ticker_list)

date_range = st.sidebar.date_input(
    "Select Date Range",
    value=(datetime.now() - timedelta(days=365*2), datetime.now()),
    max_value=datetime.now()
)

window_size = st.sidebar.slider("Prediction Window (Days)", 30, 90, 60)

# --- Initialization ---
processor = DataProcessor()
trainer = ModelTrainer(ticker=ticker)

@st.cache_data
def load_and_process(ticker):
    df = processor.load_data(ticker)
    df = processor.add_features(df)
    return df

with st.spinner(f"Fetching {ticker} market data..."):
    try:
        df_full = load_and_process(ticker)
        start_date, end_date = pd.to_datetime(date_range[0]), pd.to_datetime(date_range[1])
        df = df_full[(df_full.index >= start_date) & (df_full.index <= end_date)]
    except Exception as e:
        st.error(f"Failed to load data: {e}")
        st.stop()

# --- Top Metrics ---
current_price = df['close'].iloc[-1]
prev_price = df['close'].iloc[-2]
price_change = current_price - prev_price
pct_change = (price_change / prev_price) * 100

col1, col2, col3, col4 = st.columns(4)
with col1:
    st.metric("Current Price", f"₹{current_price:,.2f}", f"{pct_change:+.2f}%")
with col2:
    st.metric("52W High", f"₹{df['high'].tail(252).max():,.2f}")
with col3:
    st.metric("52W Low", f"₹{df['low'].tail(252).min():,.2f}")
with col4:
    st.metric("Avg Volume (30D)", f"{df['volume'].tail(30).mean()/1e6:.1f}M")

# --- Tabs ---
tab1, tab2, tab3 = st.tabs(["📈 Market Overview", "🔍 Technical Analysis", "🔮 AI Forecasting"])

with tab1:
    fig = make_subplots(rows=2, cols=1, shared_xaxes=True, 
                       vertical_spacing=0.03, row_width=[0.2, 0.7])
    fig.add_trace(go.Candlestick(
        x=df.index, open=df['open'], high=df['high'], low=df['low'], close=df['close'],
        name="Price Action"
    ), row=1, col=1)
    fig.add_trace(go.Bar(x=df.index, y=df['volume'], name="Volume", opacity=0.3), row=2, col=1)
    fig.update_layout(height=600, xaxis_rangeslider_visible=False, template="plotly_dark", margin=dict(l=20, r=20, t=50, b=20))
    st.plotly_chart(fig, use_container_width=True)

with tab2:
    col_ta1, col_ta2 = st.columns([2, 1])
    with col_ta1:
        fig_ta = go.Figure()
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['close'], name='Close', line=dict(color='white', width=1)))
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['SMA_20'], name='SMA 20', line=dict(color='cyan', width=1.5)))
        fig_ta.add_trace(go.Scatter(x=df.index, y=df['SMA_50'], name='SMA 50', line=dict(color='orange', width=1.5)))
        fig_ta.update_layout(title="Moving Averages", template="plotly_dark", height=400)
        st.plotly_chart(fig_ta, use_container_width=True)
        
        fig_rsi = go.Figure()
        fig_rsi.add_trace(go.Scatter(x=df.index, y=df['RSI'], name='RSI', line=dict(color='purple')))
        fig_rsi.add_hline(y=70, line_dash="dash", line_color="red")
        fig_rsi.add_hline(y=30, line_dash="dash", line_color="green")
        fig_rsi.update_layout(title="Relative Strength Index (RSI)", template="plotly_dark", height=250, yaxis_range=[0, 100])
        st.plotly_chart(fig_rsi, use_container_width=True)
    with col_ta2:
        st.subheader("Key Indicators")
        rsi_val = df['RSI'].iloc[-1]
        st.write(f"**RSI (14):** {rsi_val:.2f}")
        macd_val = df['MACD'].iloc[-1]
        signal_val = df['Signal_Line'].iloc[-1]
        st.write(f"**MACD:** {macd_val:.4f}")
        if macd_val > signal_val: st.success("📈 Bullish")
        else: st.error("📉 Bearish")

with tab3:
    st.subheader("AI Price Forecasting")
    
    # Status Message
    if not TENSORFLOW_AVAILABLE:
        st.warning("⚠️ TensorFlow is not supported on Python 3.14. Using **Random Forest Fallback**.")
    
    model, model_type = trainer.load_model()
    
    if model:
        st.info(f"Model Active: **{model_type}**")
        X, y, scaled_data = processor.prepare_lstm_data(df_full, window_size=window_size)
        split = int(len(X) * 0.9)
        X_test, y_test = X[split:], y[split:]
        
        with st.spinner("Generating forecasts..."):
            preds = trainer.predict(model, model_type, X_test)
            preds = processor.scaler.inverse_transform(preds)
            actual = processor.scaler.inverse_transform(y_test.reshape(-1, 1))
            
            # Future Prediction
            last_window = scaled_data[-window_size:]
            last_window = np.reshape(last_window, (1, window_size, 1))
            next_pred = trainer.predict(model, model_type, last_window)
            next_price = processor.scaler.inverse_transform(next_pred)[0][0]
            
            st.metric("Predicted Next Close", f"₹{next_price:,.2f}", f"{((next_price-current_price)/current_price)*100:+.2f}%")
            
            fig_p = go.Figure()
            test_dates = df_full.index[split + window_size:]
            fig_p.add_trace(go.Scatter(x=test_dates, y=actual.flatten(), name='Actual Price', line=dict(color='#4e79a7')))
            fig_p.add_trace(go.Scatter(x=test_dates, y=preds.flatten(), name='Predicted Price', line=dict(color='#f28e2b', dash='dash')))
            fig_p.update_layout(title="Prediction vs Actual", template="plotly_dark", height=450)
            st.plotly_chart(fig_p, use_container_width=True)
    else:
        st.error("No model found.")
        if st.button(f"Train Fallback Model for {ticker} (Local)"):
            with st.spinner("Training Random Forest model..."):
                X, y, _ = processor.prepare_lstm_data(df_full, window_size=window_size)
                trainer.train_fallback(X, y)
                st.success("Fallback model trained! Please refresh.")
                st.rerun()

st.sidebar.markdown("---")
# Branding removed
