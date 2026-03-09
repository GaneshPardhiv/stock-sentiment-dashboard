import plotly.graph_objects as go
import plotly.express as px
import pandas as pd


def plot_price_chart(df, ticker):
    """
    Candlestick + Moving Average chart for historical prices.
    """
    df = df.copy()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["MA_50"] = df["Close"].rolling(window=50).mean()

    fig = go.Figure()

    # Candlestick
    fig.add_trace(go.Candlestick(
        x=df.index,
        open=df["Open"],
        high=df["High"],
        low=df["Low"],
        close=df["Close"],
        name="Price",
        increasing_line_color="#26a69a",
        decreasing_line_color="#ef5350"
    ))

    # MA Lines
    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_20"],
        mode="lines", name="MA 20",
        line=dict(color="#FFA726", width=1.5)
    ))

    fig.add_trace(go.Scatter(
        x=df.index, y=df["MA_50"],
        mode="lines", name="MA 50",
        line=dict(color="#42A5F5", width=1.5)
    ))

    fig.update_layout(
        title=f"{ticker} — Historical Price (Last 6 Months)",
        xaxis_title="Date",
        yaxis_title="Price (USD)",
        xaxis_rangeslider_visible=False,
        template="plotly_dark",
        height=480,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def plot_prediction_chart(df, forecast_df, ticker):
    """
    Line chart showing last 30 days of actual prices + future forecast.
    """
    recent = df["Close"].tail(30)

    fig = go.Figure()

    # Actual prices
    fig.add_trace(go.Scatter(
        x=recent.index,
        y=recent.values,
        mode="lines+markers",
        name="Actual Price",
        line=dict(color="#42A5F5", width=2),
        marker=dict(size=4)
    ))

    # Forecast
    fig.add_trace(go.Scatter(
        x=forecast_df.index,
        y=forecast_df["Predicted"],
        mode="lines+markers",
        name="Predicted Price",
        line=dict(color="#66BB6A", width=2, dash="dash"),
        marker=dict(size=6, symbol="diamond")
    ))

    # Vertical line at transition point
    fig.add_vline(
        x=str(df.index[-1]),
        line_width=1,
        line_dash="dot",
        line_color="gray",
        annotation_text="Forecast Start",
        annotation_position="top right"
    )

    fig.update_layout(
        title=f"{ticker} — Price Prediction",
        xaxis_title="Date",
        yaxis_title="Price",
        template="plotly_dark",
        height=460,
        legend=dict(orientation="h", yanchor="bottom", y=1.02)
    )

    return fig


def plot_sentiment_bar(results):
    """Bar chart of individual headline sentiment scores."""
    labels = [r["headline"][:40] + "..." for r in results]
    scores = [r["score"] for r in results]
    colors = ["#66BB6A" if s > 0.1 else ("#ef5350" if s < -0.1 else "#FFA726") for s in scores]

    fig = go.Figure(go.Bar(
        x=scores,
        y=labels,
        orientation="h",
        marker_color=colors
    ))

    fig.update_layout(
        title="Sentiment Score per Headline",
        xaxis_title="Sentiment Score",
        template="plotly_dark",
        height=400
    )

    return fig
