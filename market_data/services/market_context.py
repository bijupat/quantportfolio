import logging
import pandas as pd
from datetime import date
from typing import Iterable
from django.db import transaction
from django.db.models import QuerySet

from core.models import Symbol
from market_data.models import PriceBar, MarketContext
from market_data.services.prices import get_price_bars, dataframe_from_bars

logger = logging.getLogger(__name__)

# Indices whose regime features get joined onto every symbol's training/inference
# feature matrix. Restores the two-index contract from the original
# data_loader.py::build_market_context() (see MARKET_CONTEXT_COLS in
# market_data/services/indicators.py) — a single get_market_context() call
# only ever covers one index, so both must be combined explicitly to reproduce it.
MARKET_INDICES = ["^NSEI", "^BSESN"]


def _safe_index_name(index_ticker: str) -> str:
    """Converts a Yahoo Finance index ticker into a safe column-name prefix.

    E.g. '^NSEI' -> 'NSEI', '^BSESN' -> 'BSESN'. Mirrors the `safe` variable
    from the original data_loader.py::build_market_context().
    """
    return index_ticker.replace("^", "").replace(" ", "_")


def build_market_context_math(index_df: pd.DataFrame, index_ticker: str) -> pd.DataFrame:
    """Replicates the math from build_market_context in data_loader.py.

    Column names here are intentionally unprefixed ('ret1', not 'NSEI_ret1') —
    each row is already scoped to a single index_ticker in the DB (see
    MarketContext.index_ticker), so prefixing happens one layer up, in
    get_market_context_features(), where multiple indices get combined
    into one wide feature frame and would otherwise collide on these names.
    """
    sub = index_df[["Close"]].copy()

    sub["ret1"]  = sub["Close"].pct_change(1)
    sub["ret5"]  = sub["Close"].pct_change(5)
    sub["ret20"] = sub["Close"].pct_change(20)

    sma20 = sub["Close"].rolling(20).mean()
    sub["above_sma"] = (sub["Close"] > sma20).astype(float)
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
    index_ticker is usually "^NSEI" (NIFTY 50) or "^BSESN" (SENSEX).
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
        symbol, _ = Symbol.objects.get_or_create(ticker=index_ticker)

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


def get_market_context_features(
    start: date,
    end: date,
    indices: Iterable[str] = MARKET_INDICES,
) -> pd.DataFrame:
    """Builds one wide, prefixed market-context feature frame across multiple indices.

    Calls get_market_context() per index_ticker (each index still stores its
    own DB rows under its own index_ticker with plain column names), then
    renames each index's columns with its safe-name prefix before combining
    them side by side — e.g. 'ret1' from "^NSEI" becomes 'NSEI_ret1', 'ret1'
    from "^BSESN" becomes 'BSESN_ret1' — so the result lines up exactly with
    MARKET_CONTEXT_COLS in market_data/services/indicators.py and can be
    joined directly onto a symbol's feature DataFrame.

    Args:
        start: Start date for the feature range.
        end: End date for the feature range.
        indices: Index tickers to include, defaults to MARKET_INDICES.

    Returns:
        DataFrame indexed by Date with prefixed columns for every requested
        index, or an empty DataFrame if no index had any data.
    """
    frames = []
    for index_ticker in indices:
        qs = get_market_context(index_ticker, start, end)
        data = list(qs.values("date", "values"))
        if not data:
            continue

        df = pd.DataFrame([{"Date": pd.to_datetime(d["date"]), **d["values"]} for d in data])
        df.set_index("Date", inplace=True)

        safe = _safe_index_name(index_ticker)
        df = df.rename(columns={col: f"{safe}_{col}" for col in df.columns})
        frames.append(df)

    if not frames:
        return pd.DataFrame()

    return pd.concat(frames, axis=1)