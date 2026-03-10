import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta

try:
    import yfinance as yf
    YFINANCE_AVAILABLE = True
except ImportError:
    YFINANCE_AVAILABLE = False


def load_stock_data(ticker, period="6mo"):
    """
    Load real stock data using yfinance API.
    Falls back to simulated data if yfinance is unavailable.
    """
    if YFINANCE_AVAILABLE:
        try:
            stock = yf.Ticker(ticker)
            df = stock.history(period=period)
            if df.empty:
                raise ValueError("No data returned")
            df.index = df.index.tz_localize(None)
            df = df[["Open", "High", "Low", "Close", "Volume"]]
            return df
        except Exception as e:
            print(f"yfinance error: {e} — using simulated data")
    return _simulate_stock_data(ticker)


def _simulate_stock_data(ticker, days=180):
    BASE_PRICES = {
        "AAPL": 182.0, "GOOGL": 175.0, "MSFT": 415.0,
        "AMZN": 192.0, "TSLA": 245.0, "RELIANCE": 2850.0,
        "TCS": 3920.0, "INFY": 1760.0,
    }
    np.random.seed(hash(ticker) % 1000)
    base = BASE_PRICES.get(ticker, 100.0)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")
    prices = [base]
    for _ in range(len(dates) - 1):
        prices.append(round(prices[-1] * (1 + np.random.normal(0.0005, 0.015)), 2))
    return pd.DataFrame({
        "Open":   [round(p * (1 + np.random.normal(0, 0.005)), 2) for p in prices],
        "High":   [round(p * (1 + abs(np.random.normal(0, 0.008))), 2) for p in prices],
        "Low":    [round(p * (1 - abs(np.random.normal(0, 0.008))), 2) for p in prices],
        "Close":  prices,
        "Volume": [int(np.random.randint(5_000_000, 50_000_000)) for _ in prices],
    }, index=dates)


def add_features(df):
    df = df.copy()
    df["MA_5"]         = df["Close"].rolling(window=5).mean()
    df["MA_20"]        = df["Close"].rolling(window=20).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"]   = df["Daily_Return"].rolling(window=10).std()
    df["Day_Num"]      = range(len(df))
    df.dropna(inplace=True)
    return df


def predict_prices(df, forecast_days=7):
    df_feat = add_features(df)
    feature_cols = ["Day_Num", "MA_5", "MA_20", "Volatility"]
    X = df_feat[feature_cols].values
    y = df_feat["Close"].values
    split = int(len(X) * 0.8)
    model = LinearRegression()
    model.fit(X[:split], y[:split])
    mae = mean_absolute_error(y[split:], model.predict(X[split:]))
    r2  = r2_score(y[split:], model.predict(X[split:]))
    last = df_feat.iloc[-1]
    future_dates = pd.date_range(start=df.index[-1] + timedelta(days=1), periods=forecast_days, freq="B")
    preds = []
    for i in range(forecast_days):
        X_f = [[last["Day_Num"] + i + 1, last["MA_5"], last["MA_20"], last["Volatility"]]]
        preds.append(round(model.predict(X_f)[0] + np.random.normal(0, mae * 0.1), 2))
    forecast_df = pd.DataFrame({"Predicted": preds}, index=future_dates)
    forecast_df.index.name = "Date"
    return forecast_df, round(mae, 4), round(r2, 4)
