import re

# Simple keyword-based sentiment lexicon
# In a real project this would use VADER or a transformer model
POSITIVE_WORDS = [
    "strong", "beat", "beats", "upgrade", "buy", "growth", "profit", "gains",
    "positive", "optimism", "optimistic", "surge", "rally", "record", "high",
    "expand", "partnership", "launch", "innovation", "excited", "excitement",
    "outperform", "revenue", "earnings", "rise", "rises", "rose", "up",
    "success", "successful", "good", "great", "excellent", "boost", "boosts"
]

NEGATIVE_WORDS = [
    "miss", "misses", "missed", "dip", "dips", "selloff", "sell-off", "concern",
    "concerns", "scrutiny", "regulatory", "cautious", "caution", "decline",
    "declines", "fell", "fall", "falls", "drop", "drops", "low", "loss",
    "losses", "weak", "warning", "risk", "risks", "volatile", "volatility",
    "lawsuit", "fine", "penalty", "cut", "cuts", "downgrade", "bearish"
]


def score_headline(text):
    """
    Score a single headline.
    Returns a float between -1.0 and +1.0
    """
    text_lower = text.lower()
    words = re.findall(r'\b\w+\b', text_lower)

    pos_count = sum(1 for w in words if w in POSITIVE_WORDS)
    neg_count = sum(1 for w in words if w in NEGATIVE_WORDS)

    total = pos_count + neg_count
    if total == 0:
        return 0.0  # neutral

    score = (pos_count - neg_count) / total
    return round(score, 4)


def get_sentiment_label(score):
    """Convert numeric score to label"""
    if score > 0.1:
        return "Positive"
    elif score < -0.1:
        return "Negative"
    else:
        return "Neutral"


def analyze_headlines(headlines):
    """
    Analyze a list of headlines and return structured results.
    Each result contains: headline, score, label
    """
    results = []
    for h in headlines:
        score = score_headline(h)
        label = get_sentiment_label(score)
        results.append({
            "headline": h,
            "score": score,
            "label": label
        })
    return results


def get_overall_sentiment(headlines):
    """
    Get aggregate sentiment across all headlines.
    Returns: avg_score, label, counts
    """
    results = analyze_headlines(headlines)
    avg_score = sum(r["score"] for r in results) / len(results)
    label = get_sentiment_label(avg_score)

    counts = {
        "Positive": sum(1 for r in results if r["label"] == "Positive"),
        "Negative": sum(1 for r in results if r["label"] == "Negative"),
        "Neutral": sum(1 for r in results if r["label"] == "Neutral"),
    }

    return round(avg_score, 4), label, counts
