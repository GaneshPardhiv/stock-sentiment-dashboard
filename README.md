# Real-Time Stock Sentiment + Price Prediction Dashboard

A Python-based interactive dashboard that combines financial news sentiment analysis with machine learning-based stock price prediction. Built using Streamlit and Scikit-learn.

---

## What it does

- Select any stock (AAPL, TSLA, MSFT, RELIANCE, TCS, etc.)
- Analyzes recent news headlines using keyword-based NLP sentiment scoring
- Displays overall sentiment (Positive / Negative / Neutral) with per-headline breakdown
- Shows historical candlestick chart with 20-day and 50-day moving averages
- Predicts the next N days of stock prices using a Linear Regression model trained on historical price features

---

## Tech Stack

- **Python 3.10+**
- **Streamlit** — interactive web dashboard
- **Scikit-learn** — Linear Regression model for price forecasting
- **Plotly** — interactive candlestick and line charts
- **Pandas / NumPy** — data processing and feature engineering
- **Custom NLP module** — keyword-based sentiment scoring

---

## Project Structure

```
stock-dashboard/
│
├── app.py                        # Main Streamlit app
│
├── utils/
│   ├── sentiment_analyzer.py     # NLP sentiment scoring logic
│   ├── price_predictor.py        # ML model + feature engineering
│   └── visualizer.py             # Plotly chart functions
│
├── data/                         # Placeholder for cached data
├── requirements.txt
└── .gitignore
```

---

## How to Run

**1. Clone the repository**
```bash
git clone https://github.com/your-username/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Launch the dashboard**
```bash
streamlit run app.py
```

Opens at `http://localhost:8501`

---

## How It Works

### Sentiment Analysis
News headlines are scored using a custom keyword lexicon. Each headline receives a score from **-1.0 (negative)** to **+1.0 (positive)**. The overall sentiment is an average across all headlines.

### Price Prediction
A **Linear Regression** model is trained on the following features derived from historical prices:
- Day number (time index)
- 5-day moving average (MA_5)
- 20-day moving average (MA_20)
- 10-day rolling volatility

The model is trained on 80% of historical data and evaluated on the remaining 20%, reporting MAE and R² score.

### Final Score Formula
```
Sentiment Score = (Positive - Negative) / (Positive + Negative)
Forecast = LinearRegression(Day_Num, MA_5, MA_20, Volatility)
```

---

## Dashboard Preview

| Feature | Description |
|---------|-------------|
| 📰 Sentiment Tab | Headline-by-headline NLP scoring with color-coded results |
| 📊 Price Chart | Interactive candlestick with MA overlays |
| 🔮 Prediction Tab | ML forecast chart + model metrics (MAE, R²) |

---

## Future Improvements

- Integrate live data via `yfinance` or Alpha Vantage API
- Replace keyword NLP with VADER or FinBERT for better accuracy
- Add LSTM/time-series model for improved forecasting
- Send email alerts when sentiment drops below threshold

---

## Disclaimer

> This project is for **educational purposes only**. It does not constitute financial advice. Do not make real investment decisions based on this tool.

---

## Author

**Ganesh Pardhiv Duvvuri**  
B.Tech CSE — Lovely Professional University  
[LinkedIn](https://linkedin.com/in/ganesh-pardhiv-866575252/) | [GitHub](https://github.com/GaneshPardhiv)
