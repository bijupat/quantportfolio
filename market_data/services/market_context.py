import logging
import pandas as pd
from datetime import date
from django.db import transaction
from django.db.models import QuerySet

from core.models import Symbol
from market_data.models import PriceBar, MarketContext
from market_data.services.prices import get_price_bars, dataframe_from_bars

logger = logging.getLogger(__name__)

def build_market_context_math(index_df: pd.DataFrame, index_ticker: str) -> pd.DataFrame:
    """Replicates the math from build_market_context in data_loader.py"""
    sub = index_df[["Close"]].copy()
    
    sub["ret1"]  = sub["Close"].pct_change(1)
    sub["ret5"]  = sub["Close"].pct_change(5)
    sub["ret20"] = sub["Close"].pct_change(20)

    sma20 = sub["Close"].rolling(20).mean()
    sub["above_sma_regime"] = (sub["Close"] > sma20).astype(float)
    sub["vol20"] = sub["ret1"].rolling(20).std()

    delta = sub["Close"].diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    sub["rsi"] = 100 - 100 / (1 + rs)

    sub.drop(columns=["Close"], inplace=True)
    sub.fillna(method="ffill", inplace=True)
    sub.fillna(0, inplace=True)
    return sub

def get_market_context(index_ticker: str, start: date, end: date) -> QuerySet[MarketContext]:
    """
    Ensures market context features exist for the given date range.
    index_ticker is usually "^NSEI" (NIFTY 50)
    """
    # 1. Fetch expected dates within range
    # Note: A real implementation would filter by trading days
    dates_to_check = pd.date_range(start, end).date
    
    existing_context = set(
        MarketContext.objects.filter(index_ticker=index_ticker, date__range=(start, end))
        .values_list("date", flat=True)
    )
    
    missing_dates = set(dates_to_check) - existing_context
    
    if missing_dates:
        logger.info(f"Computing missing market context for {index_ticker}")
        
        # 2. Get the index symbol and its prices
        symbol, _ = Symbol.objects.get_or_create(ticker=index_ticker, defaults={"exchange": "NSE"})
        
        # Fetching well before the start date to allow rolling windows (e.g. 20 days) to warm up
        warmup_start = start - pd.Timedelta(days=40)
        bars = get_price_bars(symbol, warmup_start, end) 
        df = dataframe_from_bars(bars)
        
        if not df.empty:
            ctx_df = build_market_context_math(df, index_ticker)
            
            contexts_to_create = []
            for missing_date in missing_dates:
                ts = pd.Timestamp(missing_date)
                if ts in ctx_df.index:
                    row = ctx_df.loc[ts]
                    contexts_to_create.append(
                        MarketContext(
                            index_ticker=index_ticker,
                            date=missing_date,
                            values=row.to_dict()
                        )
                    )
            
            with transaction.atomic():
                MarketContext.objects.bulk_create(contexts_to_create, ignore_conflicts=True)
                
    return MarketContext.objects.filter(index_ticker=index_ticker, date__range=(start, end)).order_by("date")