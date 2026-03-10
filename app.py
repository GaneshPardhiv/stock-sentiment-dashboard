import streamlit as st
import pandas as pd
from utils.sentiment_analyzer import fetch_news_headlines, analyze_headlines, get_sentiment_label
from utils.price_predictor import predict_prices, load_stock_data
from utils.visualizer import plot_price_chart, plot_prediction_chart

st.set_page_config(page_title="Stock Sentiment & Price Predictor", layout="wide", page_icon="📈")

st.title("📈 Real-Time Stock Sentiment + Price Prediction Dashboard")
st.markdown("Combines **live financial news sentiment** (NewsAPI) with **ML-based price forecasting** (yfinance + Linear Regression).")

# ---- Sidebar ----
st.sidebar.header("⚙️ Settings")
STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA"]
selected_stock = st.sidebar.selectbox("Select Stock", STOCKS)
forecast_days  = st.sidebar.slider("Forecast Days", 3, 14, 7)
show_raw       = st.sidebar.checkbox("Show Raw Data Table", False)

st.sidebar.markdown("---")
st.sidebar.markdown("### 🔑 NewsAPI Key (Optional)")
news_api_key = st.sidebar.text_input("Paste your NewsAPI key for live news", type="password")
if not news_api_key:
    st.sidebar.info("No key? Get a free one at [newsapi.org](https://newsapi.org). Without it, curated headlines are used.")

# ---- Load live stock data ----
with st.spinner(f"Fetching live data for {selected_stock} via yfinance..."):
    df = load_stock_data(selected_stock)

# ---- Top Metrics ----
current  = round(df["Close"].iloc[-1], 2)
prev     = round(df["Close"].iloc[-2], 2)
change   = round(current - prev, 2)
change_p = round((change / prev) * 100, 2)

c1, c2, c3, c4 = st.columns(4)
c1.metric("Current Price",  f"${current}",                 f"{change_p}%")
c2.metric("52W High",       f"${round(df['Close'].max(), 2)}")
c3.metric("52W Low",        f"${round(df['Close'].min(), 2)}")
c4.metric("Avg Volume",     f"{int(df['Volume'].mean()):,}")

st.markdown("---")

tab1, tab2, tab3 = st.tabs(["📰 Sentiment Analysis", "📊 Price Chart", "🔮 Price Prediction"])

# ---- TAB 1: Sentiment ----
with tab1:
    st.subheader(f"News Sentiment for {selected_stock}")
    source_label = "Live headlines via NewsAPI" if news_api_key else "Curated headlines (add NewsAPI key for live news)"
    st.caption(f"📡 Source: {source_label}")

    with st.spinner("Fetching headlines..."):
        headlines = fetch_news_headlines(selected_stock, api_key=news_api_key if news_api_key else None)

    results  = analyze_headlines(headlines)
    avg_score = sum(r["score"] for r in results) / len(results)
    label    = get_sentiment_label(avg_score)

    col1, col2 = st.columns([1, 2])
    with col1:
        st.markdown("### Overall Sentiment")
        if label == "Positive":   st.success(f"🟢 **{label}**")
        elif label == "Negative": st.error(f"🔴 **{label}**")
        else:                     st.warning(f"🟡 **{label}**")
        st.metric("Avg Score", f"{avg_score:.3f}", help="-1 (negative) to +1 (positive)")
        pos = sum(1 for r in results if r["label"] == "Positive")
        neg = sum(1 for r in results if r["label"] == "Negative")
        neu = sum(1 for r in results if r["label"] == "Neutral")
        st.markdown(f"✅ Positive: **{pos}**  ❌ Negative: **{neg}**  ➖ Neutral: **{neu}**")

    with col2:
        st.markdown("### Headline Breakdown")
        for r in results:
            icon = "🟢" if r["label"] == "Positive" else ("🔴" if r["label"] == "Negative" else "🟡")
            st.markdown(f"{icon} **{r['label']}** ({r['score']:.2f}) — {r['headline']}")

# ---- TAB 2: Price Chart ----
with tab2:
    st.subheader(f"{selected_stock} — Live Historical Price Chart (yfinance)")
    st.caption("Data sourced in real-time via yfinance API")
    fig = plot_price_chart(df, selected_stock)
    st.plotly_chart(fig, use_container_width=True)
    if show_raw:
        st.dataframe(df.tail(30), use_container_width=True)

# ---- TAB 3: Prediction ----
with tab3:
    st.subheader(f"{selected_stock} — {forecast_days}-Day Price Forecast")
    st.caption("Model: Linear Regression trained on MA_5, MA_20, Volatility features from yfinance data")
    with st.spinner("Running prediction model..."):
        forecast_df, mae, r2 = predict_prices(df, forecast_days)

    m1, m2, m3 = st.columns(3)
    m1.metric("Forecast Start", f"${forecast_df['Predicted'].iloc[0]}")
    m2.metric("Forecast End",   f"${forecast_df['Predicted'].iloc[-1]}")
    trend = "📈 Upward" if forecast_df['Predicted'].iloc[-1] > forecast_df['Predicted'].iloc[0] else "📉 Downward"
    m3.metric("Trend", trend)
    st.markdown(f"**Model MAE:** `{mae}` &nbsp; **R² Score:** `{r2}`")

    fig2 = plot_prediction_chart(df, forecast_df, selected_stock)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 Forecast Table"):
        st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")
st.caption("Data: yfinance (prices) + NewsAPI (headlines) | Model: Scikit-learn Linear Regression | Charts: Plotly | UI: Streamlit")
