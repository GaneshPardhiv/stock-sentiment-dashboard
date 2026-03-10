import re
import os
import json
from datetime import datetime

try:
    import requests
    REQUESTS_AVAILABLE = True
except ImportError:
    REQUESTS_AVAILABLE = False

POSITIVE_WORDS = [
    "strong", "beat", "beats", "upgrade", "buy", "growth", "profit", "gains",
    "positive", "optimism", "optimistic", "surge", "rally", "record", "high",
    "expand", "partnership", "launch", "innovation", "excitement", "outperform",
    "revenue", "earnings", "rise", "rises", "rose", "up", "success", "boost"
]

NEGATIVE_WORDS = [
    "miss", "misses", "missed", "dip", "dips", "selloff", "concern", "concerns",
    "scrutiny", "regulatory", "cautious", "decline", "declines", "fell", "fall",
    "drop", "drops", "loss", "losses", "weak", "warning", "risk", "volatile",
    "lawsuit", "fine", "penalty", "cut", "downgrade", "bearish"
]

FALLBACK_HEADLINES = {
    "AAPL":  ["Apple reports record iPhone sales beating estimates",
              "Apple expands AI features across product lineup",
              "Apple faces EU regulatory scrutiny over App Store",
              "Analysts upgrade Apple stock to strong buy",
              "Apple supplier concerns weigh on investor sentiment"],
    "TSLA":  ["Tesla delivers record quarterly vehicles beating forecast",
              "Tesla faces price cut concerns amid competition",
              "Elon Musk announces new Tesla model launch",
              "Tesla misses revenue estimates in latest quarter",
              "Tesla expands Gigafactory production capacity"],
    "MSFT":  ["Microsoft Azure cloud revenue surges past expectations",
              "Microsoft Copilot AI drives strong enterprise growth",
              "Microsoft faces antitrust regulatory concerns",
              "Analysts upgrade Microsoft to outperform rating",
              "Microsoft expands partnership with OpenAI"],
    "GOOGL": ["Google ad revenue beats analyst expectations",
              "Google faces antitrust lawsuit from DOJ",
              "Google Cloud growth accelerates in latest quarter",
              "Alphabet announces new AI product launches",
              "Google misses search revenue forecast slightly"],
    "AMZN":  ["Amazon AWS profit surges driving strong earnings",
              "Amazon expands same day delivery to new markets",
              "Amazon faces worker regulation concerns",
              "Analysts raise Amazon price target on AI growth",
              "Amazon Prime membership hits record high"],
}


def fetch_news_headlines(ticker, api_key=None):
    """
    Fetch real headlines using NewsAPI.
    Falls back to curated headlines if API key not provided.
    """
    if api_key and REQUESTS_AVAILABLE:
        try:
            url = "https://newsapi.org/v2/everything"
            params = {
                "q": ticker,
                "language": "en",
                "sortBy": "publishedAt",
                "pageSize": 8,
                "apiKey": api_key
            }
            response = requests.get(url, params=params, timeout=5)
            data = response.json()
            if data.get("status") == "ok" and data.get("articles"):
                return [a["title"] for a in data["articles"] if a.get("title")][:8]
        except Exception as e:
            print(f"NewsAPI error: {e} — using fallback headlines")

    # fallback curated headlines
    return FALLBACK_HEADLINES.get(ticker, [
        f"{ticker} reports strong quarterly earnings beating expectations",
        f"Analysts upgrade {ticker} stock to buy rating",
        f"{ticker} faces regulatory scrutiny amid market concerns",
        f"{ticker} announces new product launch driving excitement",
        f"Investors cautious as {ticker} misses revenue forecast",
    ])


def score_headline(text):
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)
    pos = sum(1 for w in words if w in POSITIVE_WORDS)
    neg = sum(1 for w in words if w in NEGATIVE_WORDS)
    total = pos + neg
    return round((pos - neg) / total, 4) if total > 0 else 0.0


def get_sentiment_label(score):
    if score > 0.1:   return "Positive"
    if score < -0.1:  return "Negative"
    return "Neutral"


def analyze_headlines(headlines):
    return [{
        "headline": h,
        "score": score_headline(h),
        "label": get_sentiment_label(score_headline(h))
    } for h in headlines]


def get_overall_sentiment(headlines):
    results = analyze_headlines(headlines)
    avg = sum(r["score"] for r in results) / len(results)
    counts = {
        "Positive": sum(1 for r in results if r["label"] == "Positive"),
        "Negative": sum(1 for r in results if r["label"] == "Negative"),
        "Neutral":  sum(1 for r in results if r["label"] == "Neutral"),
    }
    return round(avg, 4), get_sentiment_label(avg), counts
