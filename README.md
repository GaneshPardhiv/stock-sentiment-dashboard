# Real-Time Stock Sentiment + Price Prediction Dashboard

An interactive Python dashboard that fetches **live stock prices via yfinance**, analyzes **financial news sentiment via NewsAPI**, and forecasts future prices using a **Scikit-learn Linear Regression model**.

---

## What it does

- Fetches **real-time stock data** (AAPL, TSLA, MSFT, GOOGL, AMZN) using **yfinance API**
- Pulls **live financial news headlines** using **NewsAPI** and scores sentiment (-1 to +1)
- Displays interactive **candlestick chart** with MA-20 and MA-50 overlays
- Predicts next N days of prices using **Linear Regression** trained on technical indicators
- Works without a NewsAPI key too — falls back to curated headlines automatically

---

## Tech Stack

| Component | Technology |
|-----------|-----------|
| Web UI | Streamlit |
| Stock Data | yfinance API |
| News Data | NewsAPI |
| ML Model | Scikit-learn (Linear Regression) |
| Charts | Plotly |
| Data Processing | Pandas, NumPy |

---

## Project Structure

```
stock-dashboard/
│
├── app.py                        # Main Streamlit app
├── utils/
│   ├── sentiment_analyzer.py     # NewsAPI + NLP sentiment scoring
│   ├── price_predictor.py        # yfinance data + ML forecasting
│   └── visualizer.py             # Plotly chart functions
├── data/                         # Cache placeholder
├── requirements.txt
└── .gitignore
```

---

## How to Run

**1. Clone the repo**
```bash
git clone https://github.com/GaneshPardhiv/stock-sentiment-dashboard.git
cd stock-sentiment-dashboard
```

**2. Install dependencies**
```bash
pip install -r requirements.txt
```

**3. Run the app**
```bash
streamlit run app.py
```

**4. (Optional) Add NewsAPI key**
- Sign up free at [newsapi.org](https://newsapi.org)
- Paste your key in the sidebar input inside the app
- Without it, the app still works with curated headlines

---

## How It Works

### Live Data (yfinance)
```python
import yfinance as yf
stock = yf.Ticker("AAPL")
df = stock.history(period="6mo")
```

### News Sentiment (NewsAPI)
Headlines are fetched for the selected ticker and scored using a keyword-based NLP lexicon. Score range: **-1.0 (negative) → +1.0 (positive)**

### Price Prediction (Linear Regression)
Features used:
- Day number (time index)
- 5-day moving average (MA_5)
- 20-day moving average (MA_20)
- 10-day rolling volatility

Trained on 80% historical data, tested on 20%, reports MAE and R².

---

## Future Improvements
- Replace keyword NLP with FinBERT for better financial sentiment accuracy
- Add LSTM model for time-series forecasting
- Email/SMS alerts when sentiment drops sharply
- Support for Indian stocks (NSE/BSE via yfinance suffix e.g. TCS.NS)

---

## Disclaimer
> For **educational purposes only**. Not financial advice.

---

## Author
**Ganesh Pardhiv Duvvuri** — B.Tech CSE, Lovely Professional University  
[LinkedIn](https://linkedin.com/in/ganesh-pardhiv-866575252/) | [GitHub](https://github.com/GaneshPardhiv)
