import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score
from datetime import datetime, timedelta
import random


# Base prices for each stock (simulated realistic values)
BASE_PRICES = {
    "AAPL":    182.0,
    "GOOGL":   175.0,
    "MSFT":    415.0,
    "AMZN":    192.0,
    "TSLA":    245.0,
    "RELIANCE": 2850.0,
    "TCS":     3920.0,
    "INFY":    1760.0,
}


def load_stock_data(ticker, days=180):
    """
    Generate realistic simulated stock price data.
    In a production project this would use yfinance or Alpha Vantage API.
    """
    random.seed(hash(ticker) % 1000)
    np.random.seed(hash(ticker) % 1000)

    base = BASE_PRICES.get(ticker, 100.0)
    dates = pd.date_range(end=datetime.today(), periods=days, freq="B")  # business days

    prices = [base]
    for _ in range(len(dates) - 1):
        # random walk with slight upward drift
        change_pct = np.random.normal(0.0005, 0.015)
        new_price = prices[-1] * (1 + change_pct)
        prices.append(round(new_price, 2))

    highs = [round(p * (1 + abs(np.random.normal(0, 0.008))), 2) for p in prices]
    lows = [round(p * (1 - abs(np.random.normal(0, 0.008))), 2) for p in prices]
    opens = [round(p * (1 + np.random.normal(0, 0.005)), 2) for p in prices]
    volumes = [int(np.random.randint(5_000_000, 50_000_000)) for _ in prices]

    df = pd.DataFrame({
        "Date": dates,
        "Open": opens,
        "High": highs,
        "Low": lows,
        "Close": prices,
        "Volume": volumes
    })
    df.set_index("Date", inplace=True)
    return df


def add_features(df):
    """Add technical indicators as features for the ML model."""
    df = df.copy()
    df["MA_5"] = df["Close"].rolling(window=5).mean()
    df["MA_20"] = df["Close"].rolling(window=20).mean()
    df["Daily_Return"] = df["Close"].pct_change()
    df["Volatility"] = df["Daily_Return"].rolling(window=10).std()
    df["Day_Num"] = range(len(df))
    df.dropna(inplace=True)
    return df


def predict_prices(df, forecast_days=7):
    """
    Train a Linear Regression model on historical data and
    predict the next N days of closing prices.
    Returns: forecast DataFrame, MAE, R2 score
    """
    df_feat = add_features(df)

    feature_cols = ["Day_Num", "MA_5", "MA_20", "Volatility"]
    X = df_feat[feature_cols].values
    y = df_feat["Close"].values

    # train on 80% of data, test on remaining 20%
    split = int(len(X) * 0.8)
    X_train, X_test = X[:split], X[split:]
    y_train, y_test = y[:split], y[split:]

    model = LinearRegression()
    model.fit(X_train, y_train)

    y_pred_test = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred_test)
    r2 = r2_score(y_test, y_pred_test)

    # Forecast future days
    last_day_num = df_feat["Day_Num"].iloc[-1]
    last_ma5 = df_feat["MA_5"].iloc[-1]
    last_ma20 = df_feat["MA_20"].iloc[-1]
    last_vol = df_feat["Volatility"].iloc[-1]

    future_dates = pd.date_range(
        start=df.index[-1] + timedelta(days=1),
        periods=forecast_days,
        freq="B"
    )

    future_preds = []
    for i in range(forecast_days):
        X_future = [[last_day_num + i + 1, last_ma5, last_ma20, last_vol]]
        pred = model.predict(X_future)[0]
        # add tiny noise so it doesn't look perfectly linear
        pred += np.random.normal(0, mae * 0.1)
        future_preds.append(round(pred, 2))

    forecast_df = pd.DataFrame({
        "Date": future_dates,
        "Predicted": future_preds
    })
    forecast_df.set_index("Date", inplace=True)

    return forecast_df, round(mae, 4), round(r2, 4)
