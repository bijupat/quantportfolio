import logging
import json
import time
from datetime import date, timedelta
from django.db import transaction
from django.db.models import QuerySet
import pandas as pd

from core.models import Symbol
from market_data.models import NewsSentiment
from django.conf import settings

# Attempt to load original sentiment logic
try:
    from news_sentiment import (
        score_headlines,
        fetch_newsapi,
        generate_synthetic_news,
        aggregate_daily_sentiment,
        NEWSAPI_LOOKBACK_DAYS,
        FINBERT_AVAILABLE,
        VADER_AVAILABLE,
    )
except ImportError:
    pass

logger = logging.getLogger(__name__)


def _business_days_only(dates: set) -> pd.DatetimeIndex:
    """Filters a set of `date` objects down to weekdays only.

    Matches generate_synthetic_news's original freq="B" business-day
    convention — weekends have no market news, so they're left out here and
    fall through to the neutral 0.0 "no_news" record in get_sentiment(),
    same as before this fix.
    """
    ts_list = sorted(pd.Timestamp(d) for d in dates)
    return pd.DatetimeIndex([t for t in ts_list if t.weekday() < 5])


def get_sentiment(symbol: Symbol, start: date, end: date) -> QuerySet[NewsSentiment]:
    """
    DB-first sentiment fetcher. Uses the original NLP logic to score headlines.

    NewsAPI's free tier only ever has articles from the trailing
    ~NEWSAPI_LOOKBACK_DAYS days, regardless of what range is requested — so
    missing dates are split into a "recent" subset (fetched from the real
    API, if NEWSAPI_KEY is configured) and an "older" subset that always
    goes straight to synthetic generation. Historical coverage no longer
    depends on whether the recent-window fetch happened to return anything,
    which previously caused years of training data to be silently zero-filled
    whenever a real API key was configured.
    """
    # 1. Check existing records
    dates_to_check = pd.date_range(start, end).date
    existing_dates = set(
        NewsSentiment.objects.filter(symbol=symbol, date__range=(start, end))
        .values_list("date", flat=True)
    )
    missing_dates = set(dates_to_check) - existing_dates

    if not missing_dates:
        return NewsSentiment.objects.filter(symbol=symbol, date__range=(start, end)).order_by("date")

    logger.info(f"Computing missing sentiment for {symbol.ticker}")

    today = date.today()
    api_horizon = today - timedelta(days=NEWSAPI_LOOKBACK_DAYS)

    recent_missing = {d for d in missing_dates if d >= api_horizon}
    historical_missing = missing_dates - recent_missing

    all_articles = []
    real_covered_dates = set()

    # 2a. Recent dates: try the real API, bounded to exactly this subset's range.
    if recent_missing:
        recent_from = min(recent_missing).strftime("%Y-%m-%d")
        recent_to = max(recent_missing).strftime("%Y-%m-%d")
        real_articles = fetch_newsapi(symbol.ticker, from_date=recent_from, to_date=recent_to)

        if real_articles:
            all_articles.extend(real_articles)
            real_covered_dates = {
                pd.Timestamp(a["date"]).date() for a in real_articles
            } & recent_missing

        # Any recent date the API didn't actually cover still needs a value —
        # fall back to synthetic for just that gap, rather than the old
        # behavior of leaving it to the "no_news" 0.0 default below.
        uncovered_recent = recent_missing - real_covered_dates
        if uncovered_recent:
            all_articles.extend(
                generate_synthetic_news(symbol.ticker, _business_days_only(uncovered_recent))
            )

    # 2b. Historical dates: always synthetic — NewsAPI's free tier cannot
    # serve these regardless of whether the recent-window fetch above
    # returned real articles. This is the actual fix: previously this whole
    # branch was skipped whenever `articles` (from a single unbounded fetch)
    # was non-empty, which zero-filled nearly the entire training range.
    if historical_missing:
        all_articles.extend(
            generate_synthetic_news(symbol.ticker, _business_days_only(historical_missing))
        )

    if all_articles:
        # Score and aggregate to daily level
        daily_df = aggregate_daily_sentiment(all_articles)

        real_backend = "finbert" if FINBERT_AVAILABLE else ("vader" if VADER_AVAILABLE else "keyword")
        synthetic_backend = "synthetic_" + real_backend

        sentiments_to_create = []

        # 3. Save to database
        for missing_date in missing_dates:
            ts = pd.Timestamp(missing_date)
            backend = real_backend if missing_date in real_covered_dates else synthetic_backend

            if ts in daily_df.index:
                row = daily_df.loc[ts]
                sentiments_to_create.append(
                    NewsSentiment(
                        symbol=symbol,
                        date=missing_date,
                        sentiment_score=row.get("sentiment_score", 0.0),
                        positive_count=row.get("positive_count", 0.0),
                        negative_count=row.get("negative_count", 0.0),
                        news_volume=row.get("news_volume", 0.0),
                        backend_used=backend
                    )
                )
            else:
                # No headline (real or synthetic) landed on this exact date —
                # save a neutral 0.0 record to prevent re-fetching it on every
                # future call. Expected for weekends and low-news days.
                sentiments_to_create.append(
                    NewsSentiment(
                        symbol=symbol,
                        date=missing_date,
                        sentiment_score=0.0,
                        positive_count=0.0,
                        negative_count=0.0,
                        news_volume=0.0,
                        backend_used="no_news"
                    )
                )

        with transaction.atomic():
            NewsSentiment.objects.bulk_create(sentiments_to_create, ignore_conflicts=True)

    return NewsSentiment.objects.filter(symbol=symbol, date__range=(start, end)).order_by("date")