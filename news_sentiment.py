"""
news_sentiment.py - Financial news ingestion and sentiment scoring

Supports three modes (in order of preference):
  1. FinBERT (HuggingFace transformers) — most accurate
  2. VADER   (nltk)                     — fast, offline
  3. Rule-based keyword fallback        — always available

News sources:
  - NewsAPI  (requires NEWSAPI_KEY env var)
  - RSS feeds from financial media
  - Static keyword-based synthetic generation for demo
"""

import os
import re
import json
import time
import hashlib
import datetime
import warnings
import numpy as np
import pandas as pd
from pathlib import Path
from typing import List, Dict, Optional

warnings.filterwarnings("ignore")

from util import log, news_cache_path, NEWS_CACHE

# ─────────────────────────────────────────────
# Sentiment backend detection
# ─────────────────────────────────────────────
FINBERT_AVAILABLE = False
VADER_AVAILABLE   = False

try:
    from transformers import pipeline, AutoTokenizer, AutoModelForSequenceClassification
    FINBERT_AVAILABLE = True
    log.info("FinBERT backend available.")
except ImportError:
    log.info("transformers not installed; FinBERT unavailable.")

try:
    from nltk.sentiment.vader import SentimentIntensityAnalyzer
    import nltk
    nltk.download("vader_lexicon", quiet=True)
    VADER_AVAILABLE = True
    log.info("VADER backend available.")
except ImportError:
    log.info("nltk not installed; VADER unavailable.")

# ─────────────────────────────────────────────
# NewsAPI coverage constraint
# ─────────────────────────────────────────────
# NewsAPI's free/developer tier only serves articles from roughly the last
# month, regardless of the `from` date requested — asking further back
# doesn't error, it just silently returns nothing for the older portion.
# Callers (see market_data.services.sentiment.get_sentiment) use this to
# decide which missing dates are even worth a real API call vs. going
# straight to synthetic generation.
NEWSAPI_LOOKBACK_DAYS = 30

# ─────────────────────────────────────────────
# Bullish / Bearish keyword lexicon
# ─────────────────────────────────────────────
BULLISH_WORDS = [
    "profit", "growth", "surge", "bullish", "record", "beat", "strong",
    "expand", "upgrade", "rally", "gain", "buy", "positive", "exceed",
    "outperform", "buyback", "dividend", "acquisition", "innovative",
]
BEARISH_WORDS = [
    "loss", "decline", "bearish", "miss", "weak", "cut", "downgrade",
    "fall", "drop", "sell", "negative", "below", "underperform",
    "fraud", "probe", "penalty", "recall", "strike", "bankruptcy",
]


def keyword_sentiment(text: str) -> float:
    """Simple keyword-based sentiment: +1 bullish, -1 bearish."""
    text_lower = text.lower()
    bull = sum(1 for w in BULLISH_WORDS if w in text_lower)
    bear = sum(1 for w in BEARISH_WORDS if w in text_lower)
    if bull + bear == 0:
        return 0.0
    return (bull - bear) / (bull + bear)


# ─────────────────────────────────────────────
# FinBERT model (lazy loaded)
# ─────────────────────────────────────────────
_finbert_pipeline = None

def get_finbert():
    global _finbert_pipeline
    if _finbert_pipeline is None and FINBERT_AVAILABLE:
        try:
            log.info("Loading FinBERT model (first use)…")
            _finbert_pipeline = pipeline(
                "text-classification",
                model="ProsusAI/finbert",
                tokenizer="ProsusAI/finbert",
                device=-1,          # CPU
                max_length=128,
                truncation=True,
            )
        except Exception as e:
            log.warning(f"FinBERT load failed: {e}")
            _finbert_pipeline = None
    return _finbert_pipeline


def finbert_sentiment(texts: List[str]) -> List[float]:
    """Score a list of headlines with FinBERT. Returns list of scores in [-1, 1]."""
    pipe = get_finbert()
    if pipe is None:
        return [0.0] * len(texts)

    label_map = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
    scores = []
    for txt in texts:
        try:
            res = pipe(txt[:512])[0]
            scores.append(label_map.get(res["label"].lower(), 0.0) * res["score"])
        except Exception:
            scores.append(0.0)
    return scores


# ─────────────────────────────────────────────
# VADER scorer
# ─────────────────────────────────────────────
_vader = None

def get_vader():
    global _vader
    if _vader is None and VADER_AVAILABLE:
        _vader = SentimentIntensityAnalyzer()
    return _vader


def vader_sentiment(texts: List[str]) -> List[float]:
    sid = get_vader()
    if sid is None:
        return [0.0] * len(texts)
    return [sid.polarity_scores(t)["compound"] for t in texts]


# ─────────────────────────────────────────────
# Unified scorer
# ─────────────────────────────────────────────
def score_headlines(headlines: List[str]) -> List[float]:
    """
    Score headlines using best available backend.
    Returns scores in [-1, +1].
    """
    if not headlines:
        return []
    if FINBERT_AVAILABLE:
        return finbert_sentiment(headlines)
    elif VADER_AVAILABLE:
        return vader_sentiment(headlines)
    else:
        return [keyword_sentiment(h) for h in headlines]


# ─────────────────────────────────────────────
# NewsAPI fetcher
# ─────────────────────────────────────────────
def fetch_newsapi(
    symbol: str,
    from_date: Optional[str] = None,
    to_date: Optional[str] = None,
    days: int = 30,
) -> List[Dict]:
    """
    Fetch articles from NewsAPI within an explicit [from_date, to_date] window.
    Requires NEWSAPI_KEY env var. Returns list of {date, headline, source}.

    Args:
        symbol: Ticker, e.g. 'RELIANCE.NS'.
        from_date: 'YYYY-MM-DD' lower bound (inclusive). If omitted, defaults to
            `days` before `to_date` — preserves the original days-back behavior
            for existing callers that don't pass explicit bounds.
        to_date: 'YYYY-MM-DD' upper bound (inclusive). Defaults to today.
        days: Fallback window size (days back from to_date), used only when
            from_date is not supplied.

    Note: NewsAPI's free/developer tier only actually has articles from the
    last ~NEWSAPI_LOOKBACK_DAYS days — requesting further back won't error,
    it will just return no matching articles for the older portion of the
    range. Callers needing full historical coverage should pair this with
    generate_synthetic_news() for dates outside that window rather than
    relying on this function alone.
    """
    api_key = os.environ.get("NEWSAPI_KEY", "")
    if not api_key:
        return []

    try:
        import requests
        query = symbol.replace(".NS", "").replace(".BO", "")

        to_dt_obj = (
            datetime.datetime.strptime(to_date, "%Y-%m-%d")
            if to_date else datetime.datetime.today()
        )
        from_dt = from_date or (to_dt_obj - datetime.timedelta(days=days)).strftime("%Y-%m-%d")
        to_dt = to_dt_obj.strftime("%Y-%m-%d")

        url = (
            f"https://newsapi.org/v2/everything"
            f"?q={query}+stock+india"
            f"&from={from_dt}&to={to_dt}&sortBy=publishedAt"
            f"&language=en&pageSize=100"
            f"&apiKey={api_key}"
        )
        resp = requests.get(url, timeout=10)
        if resp.status_code != 200:
            return []
        articles = resp.json().get("articles", [])
        return [
            {
                "date":     a["publishedAt"][:10],
                "headline": a["title"] or "",
                "source":   a.get("source", {}).get("name", "unknown"),
            }
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        log.warning(f"NewsAPI error for {symbol}: {e}")
        return []


# ─────────────────────────────────────────────
# Synthetic headline generator (demo / fallback)
# ─────────────────────────────────────────────
def generate_synthetic_news(symbol: str, dates: pd.DatetimeIndex) -> List[Dict]:
    """
    Create plausible synthetic financial headlines with random sentiment.
    Used when no real news API is available.
    """
    rng = np.random.default_rng(int(hashlib.md5(symbol.encode()).hexdigest(), 16) % 2**32)
    name = symbol.replace(".NS", "").replace(".BO", "")

    templates_bull = [
        f"{name} beats quarterly earnings estimates",
        f"{name} announces expansion plans",
        f"Analysts upgrade {name} with higher price target",
        f"{name} reports record revenue",
        f"Investors bullish on {name} outlook",
    ]
    templates_bear = [
        f"{name} misses analyst expectations",
        f"Regulatory concerns weigh on {name}",
        f"{name} faces margin pressure",
        f"Weak guidance from {name} management",
        f"Sector headwinds impact {name}",
    ]
    templates_neutral = [
        f"{name} holds annual general meeting",
        f"{name} appoints new board member",
        f"Analyst initiates coverage on {name}",
    ]

    articles = []
    for dt in dates:
        if rng.random() < 0.4:          # 40% chance of news on any day
            tone = rng.choice(["bull", "bear", "neutral"], p=[0.45, 0.35, 0.20])
            if tone == "bull":
                headline = rng.choice(templates_bull)
            elif tone == "bear":
                headline = rng.choice(templates_bear)
            else:
                headline = rng.choice(templates_neutral)
            articles.append({
                "date":     str(dt.date()),
                "headline": headline,
                "source":   "synthetic",
            })
    return articles


# ─────────────────────────────────────────────
# Aggregate to daily sentiment features
# ─────────────────────────────────────────────
def aggregate_daily_sentiment(articles: List[Dict]) -> pd.DataFrame:
    """
    Given a list of {date, headline} dicts, compute daily sentiment features.
    Returns DataFrame indexed by Date.
    """
    if not articles:
        return pd.DataFrame(columns=["sentiment_score", "positive_count",
                                     "negative_count", "news_volume"])

    df = pd.DataFrame(articles)
    df["date"] = pd.to_datetime(df["date"])
    df["score"] = score_headlines(df["headline"].tolist())

    daily = df.groupby("date").agg(
        sentiment_score  = ("score", "mean"),
        positive_count   = ("score", lambda x: (x > 0.1).sum()),
        negative_count   = ("score", lambda x: (x < -0.1).sum()),
        news_volume      = ("score", "count"),
    ).reset_index().rename(columns={"date": "Date"})

    daily.set_index("Date", inplace=True)
    daily.index = pd.to_datetime(daily.index)
    return daily


# ─────────────────────────────────────────────
# Main entry: fetch_news_sentiment
# ─────────────────────────────────────────────
def fetch_news_sentiment(
    symbol: str,
    start: str = "2013-01-01",
    end: Optional[str] = None,
    use_cache: bool = True,
    use_synthetic: bool = True,
) -> pd.DataFrame:
    """
    Fetch and score news for a symbol.  Returns daily sentiment DataFrame.

    Pipeline:
      1. Try cache
      2. Try NewsAPI
      3. Fall back to synthetic (if use_synthetic=True)
    """
    cache_file = news_cache_path(symbol)

    # Cache
    if use_cache and cache_file.exists():
        age_days = (time.time() - cache_file.stat().st_mtime) / 86400
        if age_days < 1:
            log.debug(f"News cache hit -> {symbol}")
            with open(cache_file) as f:
                articles = json.load(f)
            return aggregate_daily_sentiment(articles)

    # Try real API
    articles = fetch_newsapi(symbol, days=3650)

    # Synthetic fallback
    if not articles and use_synthetic:
        date_range = pd.date_range(start=start, end=end or pd.Timestamp.today(), freq="B")
        articles = generate_synthetic_news(symbol, date_range)

    # Cache
    with open(cache_file, "w") as f:
        json.dump(articles, f)

    return aggregate_daily_sentiment(articles)