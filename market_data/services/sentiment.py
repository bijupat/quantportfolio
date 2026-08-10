import logging
import json
import time
from datetime import date
from django.db import transaction
from django.db.models import QuerySet
import pandas as pd

from core.models import Symbol
from market_data.models import NewsSentiment
from django.conf import settings

# Attempt to load original sentiment logic
try:
    from news_sentiment import score_headlines, fetch_newsapi, generate_synthetic_news, aggregate_daily_sentiment
except ImportError:
    pass

logger = logging.getLogger(__name__)

def get_sentiment(symbol: Symbol, start: date, end: date) -> QuerySet[NewsSentiment]:
    """
    DB-first sentiment fetcher. Uses the original NLP logic to score headlines.
    """
    # 1. Check existing records
    dates_to_check = pd.date_range(start, end).date
    existing_dates = set(
        NewsSentiment.objects.filter(symbol=symbol, date__range=(start, end))
        .values_list("date", flat=True)
    )
    
    missing_dates = set(dates_to_check) - existing_dates
    
    if missing_dates:
        logger.info(f"Computing missing sentiment for {symbol.ticker}")
        
        # Determine the earliest date we need to fetch news for
        fetch_start_dt = min(missing_dates).strftime("%Y-%m-%d")
        
        # 2. Fetch using original news_sentiment.py logic
        # Try real API first
        articles = fetch_newsapi(symbol.ticker, days=60) # Over-fetch slightly to ensure coverage
        
        # Synthetic fallback if API fails or returns nothing
        if not articles:
            logger.info(f"Using synthetic news for {symbol.ticker}")
            dt_range = pd.date_range(start=fetch_start_dt, end=end.strftime("%Y-%m-%d"), freq="B")
            articles = generate_synthetic_news(symbol.ticker, dt_range)
            
        if articles:
            # Score and aggregate to daily level
            daily_df = aggregate_daily_sentiment(articles)
            
            # Determine backend used for logging
            from news_sentiment import FINBERT_AVAILABLE, VADER_AVAILABLE
            backend = "finbert" if FINBERT_AVAILABLE else ("vader" if VADER_AVAILABLE else "keyword")
            if not fetch_newsapi(symbol.ticker, days=1): # Check if API is actually returning
                backend = "synthetic_" + backend

            sentiments_to_create = []
            
            # 3. Save to database
            for missing_date in missing_dates:
                ts = pd.Timestamp(missing_date)
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
                    # Save a neutral 0.0 record if no news existed on this missing date to prevent re-fetching
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