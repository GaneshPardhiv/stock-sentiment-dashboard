import streamlit as st
import pandas as pd
import json
from utils.sentiment_analyzer import analyze_headlines, get_sentiment_label
from utils.price_predictor import predict_prices, load_stock_data
from utils.visualizer import plot_price_chart, plot_sentiment_gauge, plot_prediction_chart

st.set_page_config(
    page_title="Stock Sentiment & Price Predictor",
    layout="wide",
    page_icon="📈"
)

st.title("📈 Real-Time Stock Sentiment + Price Prediction Dashboard")
st.markdown("Combines financial news sentiment analysis with ML-based stock price forecasting.")

# ---- Sidebar ----
st.sidebar.header("⚙️ Dashboard Settings")

STOCKS = ["AAPL", "GOOGL", "MSFT", "AMZN", "TSLA", "RELIANCE", "TCS", "INFY"]
selected_stock = st.sidebar.selectbox("Select Stock", STOCKS)
forecast_days = st.sidebar.slider("Forecast Days", min_value=3, max_value=14, value=7)
show_raw = st.sidebar.checkbox("Show Raw Data Table", value=False)

st.sidebar.markdown("---")
st.sidebar.info("📌 Uses simulated data + real ML models for demo purposes.")

# ---- Load Data ----
with st.spinner(f"Loading data for {selected_stock}..."):
    df = load_stock_data(selected_stock)

# ---- Layout ----
col1, col2, col3, col4 = st.columns(4)

current_price = round(df["Close"].iloc[-1], 2)
prev_price = round(df["Close"].iloc[-2], 2)
change = round(current_price - prev_price, 2)
change_pct = round((change / prev_price) * 100, 2)

col1.metric("Current Price", f"${current_price}", f"{change_pct}%")
col2.metric("52W High", f"${round(df['Close'].max(), 2)}")
col3.metric("52W Low", f"${round(df['Close'].min(), 2)}")
col4.metric("Avg Volume", f"{int(df['Volume'].mean()):,}")

st.markdown("---")

# ---- Tabs ----
tab1, tab2, tab3 = st.tabs(["📰 Sentiment Analysis", "📊 Price Chart", "🔮 Price Prediction"])

# ---- TAB 1: Sentiment ----
with tab1:
    st.subheader(f"Latest News Sentiment for {selected_stock}")

    headlines = [
        f"{selected_stock} reports strong quarterly earnings, beating analyst expectations",
        f"Analysts upgrade {selected_stock} to buy rating amid market optimism",
        f"{selected_stock} stock dips on broader market selloff concerns",
        f"New product launch drives excitement around {selected_stock}",
        f"Investors cautious as {selected_stock} faces regulatory scrutiny",
        f"{selected_stock} expands into new markets with strategic partnership",
        f"CEO of {selected_stock} speaks positively about growth outlook",
        f"{selected_stock} misses revenue forecast in latest quarterly report",
    ]

    results = analyze_headlines(headlines)
    avg_score = sum(r["score"] for r in results) / len(results)
    overall_label = get_sentiment_label(avg_score)

    col_sent1, col_sent2 = st.columns([1, 2])

    with col_sent1:
        st.markdown("### Overall Sentiment")
        if overall_label == "Positive":
            st.success(f"🟢 **{overall_label}**")
        elif overall_label == "Negative":
            st.error(f"🔴 **{overall_label}**")
        else:
            st.warning(f"🟡 **{overall_label}**")

        st.metric("Sentiment Score", f"{avg_score:.3f}", help="Range: -1 (negative) to +1 (positive)")

        positive = sum(1 for r in results if r["label"] == "Positive")
        negative = sum(1 for r in results if r["label"] == "Negative")
        neutral = sum(1 for r in results if r["label"] == "Neutral")

        st.markdown(f"✅ Positive: **{positive}** headlines")
        st.markdown(f"❌ Negative: **{negative}** headlines")
        st.markdown(f"➖ Neutral: **{neutral}** headlines")

    with col_sent2:
        st.markdown("### Headline-wise Breakdown")
        for r in results:
            icon = "🟢" if r["label"] == "Positive" else ("🔴" if r["label"] == "Negative" else "🟡")
            st.markdown(f"{icon} **{r['label']}** ({r['score']:.2f}) — {r['headline']}")

# ---- TAB 2: Price Chart ----
with tab2:
    st.subheader(f"{selected_stock} — Historical Price Chart (Last 6 Months)")
    fig = plot_price_chart(df, selected_stock)
    st.plotly_chart(fig, use_container_width=True)

    if show_raw:
        st.subheader("Raw Price Data")
        st.dataframe(df.tail(30), use_container_width=True)

# ---- TAB 3: Price Prediction ----
with tab3:
    st.subheader(f"{selected_stock} — {forecast_days}-Day Price Forecast")
    st.caption("Model: Linear Regression trained on historical closing prices")

    with st.spinner("Running prediction model..."):
        forecast_df, mae, r2 = predict_prices(df, forecast_days)

    col_m1, col_m2, col_m3 = st.columns(3)
    col_m1.metric("Forecast Start", f"${round(forecast_df['Predicted'].iloc[0], 2)}")
    col_m2.metric("Forecast End", f"${round(forecast_df['Predicted'].iloc[-1], 2)}")
    trend = "📈 Upward" if forecast_df['Predicted'].iloc[-1] > forecast_df['Predicted'].iloc[0] else "📉 Downward"
    col_m3.metric("Trend", trend)

    st.markdown(f"**Model MAE:** `{mae:.2f}` &nbsp;&nbsp; **R² Score:** `{r2:.4f}`")

    fig2 = plot_prediction_chart(df, forecast_df, selected_stock)
    st.plotly_chart(fig2, use_container_width=True)

    with st.expander("📋 View Forecast Table"):
        st.dataframe(forecast_df, use_container_width=True)

st.markdown("---")
st.caption("Built with Python | Streamlit | Scikit-learn | NLP | Plotly  — For educational purposes only. Not financial advice.")
