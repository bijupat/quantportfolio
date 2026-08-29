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

# market_data/services/prices.py

# Tickers trusted to teach the SHARED NonTradingDay cache about holidays.
# These are broad indices that have existed continuously for decades, so
# "yfinance returned nothing for this date" reliably means "market closed,"
# not "this instrument didn't exist yet."
#
# Regular equity tickers must NEVER write to NonTradingDay: a stock that
# IPO'd in 2023 returns empty data for every day before its listing, and if
# that gets recorded as a global holiday, every OTHER (much older) symbol
# processed afterwards silently loses real history for that whole window.
# Confirmed via fetch_prices job on nifty100 (Aug 2026): VEDL/TATAPOWER/
# TORNTPHARM/UNIONBANK/UNITDSPR/ZYDUSLIFE all lost 11+ years of history this way.
MARKET_CALENDAR_AUTHORITIES = {"^NSEI", "^BSESN"}


def get_price_bars(symbol: Symbol, start: date, end: date) -> QuerySet[PriceBar]:
    """Ensures PriceBar rows exist for symbol across [start, end], fetching gaps from yfinance.

    Only MARKET_CALENDAR_AUTHORITIES may record entries in the shared
    NonTradingDay cache. A regular equity returning no data for a range
    usually just means it wasn't listed yet — symbol-specific, and never
    to be read as "the market was closed for everyone."

    Pre-listing gap optimization (history_confirmed_start): a newly-listed
    or recently-renamed symbol (e.g. ETERNAL.NS/Zomato, which didn't list
    until 23 Jul 2021 — see core.models.Symbol.history_confirmed_start's
    docstring) will have yfinance return an empty DataFrame for every
    pre-listing date range requested. Without this optimization, every
    single call to get_price_bars() for that symbol across a training
    window starting before its listing date re-issues the same doomed
    yfinance queries and re-triggers yfinance's own "possibly delisted"
    warning for each chunk, every run, forever — since no PriceBar rows
    exist to satisfy `existing_dates`, and (correctly) no NonTradingDay
    rows get written either, since only MARKET_CALENDAR_AUTHORITIES may do
    that. Once a fetch has confirmed where real data actually begins for
    this specific symbol, that boundary is remembered on
    symbol.history_confirmed_start, and this function raises its internal
    fetch start up to that boundary on every subsequent call — the queried
    range returned to the caller still honors whatever range they actually
    asked for (rows simply won't exist before the confirmed start, exactly
    as before this optimization), only the *yfinance* fetch range is
    narrowed.
    """
    existing_dates = set(
        PriceBar.objects.filter(symbol=symbol, date__range=(start, end))
        .values_list("date", flat=True)
    )
    known_holidays = set(NonTradingDay.objects.values_list("date", flat=True))

    trading_days = _expected_trading_days(start, end) - known_holidays
    missing_days = sorted(trading_days - existing_dates)

    is_calendar_authority = symbol.ticker in MARKET_CALENDAR_AUTHORITIES

    # Skip re-querying yfinance for any date already proven to have no data
    # for THIS symbol, per a prior confirmed start. Calendar-authority
    # tickers (^NSEI, ^BSESN) are exempted — they're the ones establishing
    # NonTradingDay in the first place and have effectively unbounded
    # history, so this optimization has nothing to save for them anyway.
    if not is_calendar_authority and symbol.history_confirmed_start is not None:
        missing_days = [d for d in missing_days if d >= symbol.history_confirmed_start]

    if missing_days:
        logger.info(f"Missing {len(missing_days)} days for {symbol.ticker}. Fetching from yfinance...")

        # Tracks the earliest date any chunk in THIS call actually returned
        # real data for — used to tighten symbol.history_confirmed_start
        # once, after the loop, rather than writing to the DB on every chunk.
        earliest_real_date_this_call: date | None = None

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
                    if returned_dates:
                        chunk_earliest = min(returned_dates)
                        if earliest_real_date_this_call is None or chunk_earliest < earliest_real_date_this_call:
                            earliest_real_date_this_call = chunk_earliest
                else:
                    returned_dates = set()

                # Only a trusted index ticker may promote "no data returned"
                # into a shared holiday record.
                if is_calendar_authority:
                    requested_days = _expected_trading_days(lo, hi)
                    confirmed_holidays = requested_days - returned_dates
                    if confirmed_holidays:
                        NonTradingDay.objects.bulk_create(
                            [NonTradingDay(date=d) for d in confirmed_holidays], ignore_conflicts=True
                        )

            except Exception as e:
                logger.error(f"Failed to fetch {symbol.ticker} from yfinance: {e}")

            time.sleep(0.5)

        # Tighten history_confirmed_start if this call discovered real data
        # earlier than previously known — never regress it to a LATER date,
        # which would silently forget an earlier confirmed start recorded
        # by a prior run (see Symbol.history_confirmed_start's docstring).
        if earliest_real_date_this_call is not None and not is_calendar_authority:
            current = symbol.history_confirmed_start
            if current is None or earliest_real_date_this_call < current:
                symbol.history_confirmed_start = earliest_real_date_this_call
                symbol.save(update_fields=["history_confirmed_start"])
                logger.info(
                    f"{symbol.ticker}: history_confirmed_start set to "
                    f"{earliest_real_date_this_call} (earliest real data found this run)."
                )

    return PriceBar.objects.filter(symbol=symbol, date__range=(start, end)).order_by("date")


def sync_market_calendar(start: date, end: date) -> int:
    """Rebuilds the shared NonTradingDay cache from a trusted index ticker.

    Call before bulk-fetching a universe of equities so legitimate holidays
    are known upfront, rather than being (unsafely) inferred per-symbol.

    Args:
        start: Start of the date range to sync.
        end: End of the date range to sync.

    Returns:
        Count of NonTradingDay rows present in [start, end] after sync.
    """
    symbol, _ = Symbol.objects.get_or_create(ticker="^NSEI")
    get_price_bars(symbol, start, end)  # writes NonTradingDay as a side effect (authority ticker)
    return NonTradingDay.objects.filter(date__range=(start, end)).count()


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