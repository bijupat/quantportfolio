import logging
import numpy as np
import pandas as pd
from datetime import date
from django.db import transaction
from django.db.models import QuerySet

from core.models import Symbol
from market_data.models import PriceBar, TechnicalIndicatorSnapshot
from market_data.services.prices import dataframe_from_bars

logger = logging.getLogger(__name__)


FEATURE_COLUMNS = [
    "returns_1d", "returns_5d", "returns_20d", "log_return",
    "hl_ratio", "co_ratio",
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma100",
    "rsi_14", "rsi_6", "stoch_rsi",
    "macd_norm", "macd_hist",
    "momentum_10", "momentum_20", "momentum_60",
    "adx", "di_plus", "di_minus",
    "atr_norm", "bb_width", "bb_pct", "hv_20", "hv_60",
    "volume_ratio", "obv_signal", "close_vs_vwap",
    "doji", "bullish_engulf",
    "day_of_week", "month", "quarter",
    "sentiment_score", "positive_count", "negative_count", "news_volume",
]

MARKET_CONTEXT_COLS = [
    "NSEI_ret1", "NSEI_ret5", "NSEI_ret20",
    "NSEI_above_sma", "NSEI_vol20", "NSEI_rsi",
    "BSESN_ret1", "BSESN_ret5", "BSESN_ret20",
    "BSESN_above_sma", "BSESN_vol20", "BSESN_rsi",
]

# --- Replicating your original indicator math ---

def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    # Price-derived
    df["returns_1d"]  = close.pct_change(1)
    df["returns_5d"]  = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)
    df["log_return"]  = np.log(close / close.shift(1))
    df["hl_ratio"]    = (high - low) / (close + 1e-9)
    df["co_ratio"]    = (close - open_) / (open_ + 1e-9)

    # Moving averages
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{w}"]  = close.rolling(w).mean()
        df[f"ema_{w}"]  = close.ewm(span=w, adjust=False).mean()

    for w in [20, 50, 100]:
        df[f"close_vs_sma{w}"] = close / (df[f"sma_{w}"] + 1e-9) - 1

    # Momentum (RSI, MACD)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    gain6 = delta.clip(lower=0).rolling(6).mean()
    loss6 = (-delta.clip(upper=0)).rolling(6).mean()
    rs6   = gain6 / (loss6 + 1e-9)
    df["rsi_6"] = 100 - 100 / (1 + rs6)

    rsi_min = df["rsi_14"].rolling(14).min()
    rsi_max = df["rsi_14"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_norm"]   = df["macd"] / (close + 1e-9)

    df["momentum_10"] = close / (close.shift(10) + 1e-9) - 1
    df["momentum_20"] = close / (close.shift(20) + 1e-9) - 1
    df["momentum_60"] = close / (close.shift(60) + 1e-9) - 1

    # Trend (ADX)
    df["tr"] = np.maximum(high - low, np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))))
    df["dm_plus"]  = np.where((high - high.shift(1)) > (low.shift(1) - low), np.maximum(high - high.shift(1), 0), 0)
    df["dm_minus"] = np.where((low.shift(1) - low) > (high - high.shift(1)), np.maximum(low.shift(1) - low, 0), 0)
    atr14      = df["tr"].rolling(14).mean()
    dip        = df["dm_plus"].rolling(14).mean()  / (atr14 + 1e-9) * 100
    dim        = df["dm_minus"].rolling(14).mean() / (atr14 + 1e-9) * 100
    dx         = abs(dip - dim) / (dip + dim + 1e-9) * 100
    df["adx"]  = dx.rolling(14).mean()
    df["di_plus"]  = dip
    df["di_minus"] = dim
    df.drop(columns=["tr", "dm_plus", "dm_minus"], inplace=True)

    # Volatility
    high_low = high - low
    high_close_prev = abs(high - close.shift(1))
    low_close_prev  = abs(low  - close.shift(1))
    tr_vals = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr_vals.rolling(14).mean()
    df["atr_norm"] = df["atr_14"] / (close + 1e-9)

    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"]  = (bb_upper - bb_lower) / (sma20 + 1e-9)
    df["bb_pct"]    = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    df["hv_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["hv_60"] = df["log_return"].rolling(60).std() * np.sqrt(252)

    # Volume
    df["volume_sma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma20"] + 1e-9)
    
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"]         = obv
    df["obv_ema20"]   = obv.ewm(span=20, adjust=False).mean()
    df["obv_signal"]  = (obv - df["obv_ema20"]) / (df["obv_ema20"].abs() + 1e-9)

    typical = (high + low + close) / 3
    df["vwap_20"]     = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    df["close_vs_vwap"] = close / (df["vwap_20"] + 1e-9) - 1

    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df.fillna(method="ffill", inplace=True)
    df.fillna(0, inplace=True) # Fallback for initial rows
    return df

def build_advanced_indicators(ohlcv: pd.DataFrame) -> pd.DataFrame:
    o = ohlcv["Open"]
    h = ohlcv["High"]
    l = ohlcv["Low"]
    c = ohlcv["Close"]
    results = {}

    results["roc_10"] = c.pct_change(10) * 100
    results["roc_20"] = c.pct_change(20) * 100

    highest_high = h.rolling(14).max()
    lowest_low   = l.rolling(14).min()
    results["williams_r_14"] = -100 * (highest_high - c) / (highest_high - lowest_low + 1e-9)

    typical = (h + l + c) / 3
    sma_tp  = typical.rolling(20).mean()
    mad     = typical.rolling(20).apply(lambda x: np.abs(x - x.mean()).mean(), raw=True)
    results["cci_20"] = (typical - sma_tp) / (0.015 * mad + 1e-9)
    
    log_hl = np.log(h / l) ** 2
    results["parkinson_vol_20"] = np.sqrt(log_hl.rolling(20).mean() / (4 * np.log(2))) * np.sqrt(252)

    log_hl_gk = 0.5 * np.log(h / l) ** 2
    log_co = (2 * np.log(2) - 1) * np.log(c / o) ** 2
    results["garman_klass_vol_20"] = np.sqrt(252 * (log_hl_gk - log_co).rolling(20).mean())

    out = pd.DataFrame(results, index=ohlcv.index)
    out.replace([np.inf, -np.inf], np.nan, inplace=True)
    out.fillna(method="ffill", inplace=True)
    out.fillna(0, inplace=True)
    return out


# --- DB-First Django Orchestrator ---

def get_indicators(symbol: Symbol, bars_qs: QuerySet[PriceBar]) -> QuerySet[TechnicalIndicatorSnapshot]:
    """
    Computes and stores missing indicators in the database.
    """
    dates_in_qs = set(bars_qs.values_list('date', flat=True))
    existing_indicators = set(
        TechnicalIndicatorSnapshot.objects.filter(symbol=symbol, date__in=dates_in_qs)
        .values_list("date", flat=True)
    )
    
    missing_dates = dates_in_qs - existing_indicators
    
    if missing_dates:
        logger.info(f"Computing missing indicators for {len(missing_dates)} days on {symbol.ticker}")
        
        # We need the full dataframe history to compute rolling values correctly (like SMA200)
        # So we fetch all available bars for this symbol to run the math
        all_bars = PriceBar.objects.filter(symbol=symbol).order_by("date")
        df = dataframe_from_bars(all_bars)
        
        if not df.empty:
            df_tech = calculate_technical_indicators(df)
            df_adv = build_advanced_indicators(df)
            
            # Combine them
            df_combined = df_tech.join(df_adv, how="left")
            df_combined = df_combined.drop(columns=["Open", "High", "Low", "Close", "Volume"])
            
            snapshots_to_create = []
            
            for missing_date in missing_dates:
                # pandas datetime indexing
                ts = pd.Timestamp(missing_date)
                if ts in df_combined.index:
                    row = df_combined.loc[ts]
                    snapshots_to_create.append(
                        TechnicalIndicatorSnapshot(
                            symbol=symbol,
                            date=missing_date,
                            values=row.to_dict()
                        )
                    )
            
            with transaction.atomic():
                TechnicalIndicatorSnapshot.objects.bulk_create(snapshots_to_create, ignore_conflicts=True)
    
    return TechnicalIndicatorSnapshot.objects.filter(symbol=symbol, date__in=dates_in_qs).order_by("date")