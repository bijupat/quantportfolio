import time
import logging
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import date, timedelta
from typing import List

from django.db import transaction
from django.db.models import QuerySet
from core.models import Symbol
from market_data.models import PriceBar, NonTradingDay

logger = logging.getLogger(__name__)

def _expected_trading_days(start: date, end: date) -> set[date]:
    """Generates a naive set of expected trading days (Monday-Friday)."""
    days = set()
    current = start
    while current <= end:
        if current.weekday() < 5:  # 0-4 are Mon-Fri
            days.add(current)
        current += timedelta(days=1)
    return days

def _contiguous_ranges(dates: List[date]) -> List[tuple[date, date]]:
    """Groups a sorted list of dates into contiguous (start, end) tuples."""
    if not dates:
        return []
    
    ranges = []
    start_date = dates[0]
    prev_date = dates[0]

    for current_date in dates[1:]:
        if current_date != prev_date + timedelta(days=1):
            ranges.append((start_date, prev_date))
            start_date = current_date
        prev_date = current_date
        
    ranges.append((start_date, prev_date))
    return ranges


def get_price_bars(symbol: Symbol, start: date, end: date) -> QuerySet[PriceBar]:
    existing_dates = set(
        PriceBar.objects.filter(symbol=symbol, date__range=(start, end))
        .values_list("date", flat=True)
    )
    known_holidays = set(NonTradingDay.objects.values_list("date", flat=True))

    trading_days = _expected_trading_days(start, end) - known_holidays
    missing_days = sorted(trading_days - existing_dates)

    if missing_days:
        logger.info(f"Missing {len(missing_days)} days for {symbol.ticker}. Fetching from yfinance...")

        for lo, hi in _contiguous_ranges(missing_days):
            yf_end = hi + timedelta(days=1)
            try:
                ticker = yf.Ticker(symbol.ticker)
                df = ticker.history(start=lo.strftime("%Y-%m-%d"), end=yf_end.strftime("%Y-%m-%d"), auto_adjust=True)

                if not df.empty:
                    df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
                    df.index = pd.to_datetime(df.index).tz_localize(None)
                    df.replace([np.inf, -np.inf], np.nan, inplace=True)
                    df.dropna(how="all", inplace=True)
                    df["Close"] = df["Close"].ffill()
                    for col in ["Open", "High", "Low"]:
                        df[col] = df[col].fillna(df["Close"])
                    df["Volume"] = df["Volume"].fillna(0)

                    bars_to_create = [
                        PriceBar(symbol=symbol, date=idx.date(), open=row["Open"],
                                 high=row["High"], low=row["Low"], close=row["Close"], volume=row["Volume"])
                        for idx, row in df.iterrows()
                    ]
                    with transaction.atomic():
                        PriceBar.objects.bulk_create(bars_to_create, ignore_conflicts=True)

                    returned_dates = set(pd.to_datetime(df.index).date)
                else:
                    returned_dates = set()

                # Any requested trading day yfinance didn't return is a confirmed holiday —
                # cache it so it's never re-queried again for ANY symbol.
                requested_days = _expected_trading_days(lo, hi)
                confirmed_holidays = requested_days - returned_dates
                if confirmed_holidays:
                    NonTradingDay.objects.bulk_create(
                        [NonTradingDay(date=d) for d in confirmed_holidays], ignore_conflicts=True
                    )

            except Exception as e:
                logger.error(f"Failed to fetch {symbol.ticker} from yfinance: {e}")

            time.sleep(0.5)

    return PriceBar.objects.filter(symbol=symbol, date__range=(start, end)).order_by("date")

def dataframe_from_bars(queryset: QuerySet[PriceBar]) -> pd.DataFrame:
    """Helper to convert the Django QuerySet back into a Pandas DataFrame for technical analysis."""
    data = list(queryset.values("date", "open", "high", "low", "close", "volume"))
    if not data:
        return pd.DataFrame()
        
    df = pd.DataFrame(data)
    df.rename(columns={
        "date": "Date", "open": "Open", "high": "High", 
        "low": "Low", "close": "Close", "volume": "Volume"
    }, inplace=True)
    df.set_index("Date", inplace=True)
    df.index = pd.to_datetime(df.index)
    
    # Ensure float types
    for col in ["Open", "High", "Low", "Close", "Volume"]:
        df[col] = df[col].astype(float)
        
    return df

def backfill_symbol_metadata(symbol: Symbol) -> bool:
    """Fetches and saves name/sector for a Symbol from yfinance if missing.

    Args:
        symbol: The Symbol instance to backfill. Mutated and saved in place.

    Returns:
        True if metadata was fetched and saved this call, False if it was
        already populated or the yfinance fetch failed/returned nothing.
    """
    if symbol.name and symbol.sector:
        return False

    yf_ticker = symbol.ticker if symbol.ticker.endswith((".NS", ".BO")) else f"{symbol.ticker}.NS"

    try:
        info = yf.Ticker(yf_ticker).info
    except Exception as e:
        logger.warning(f"yfinance .info fetch failed for {symbol.ticker}: {e}")
        return False

    if not info:
        logger.warning(f"yfinance returned empty metadata for {symbol.ticker}")
        return False

    symbol.name = symbol.name or info.get("longName") or info.get("shortName") or symbol.ticker
    symbol.sector = symbol.sector or info.get("sector") or "Unknown"
    symbol.save(update_fields=["name", "sector"])
    return True