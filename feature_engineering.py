"""
feature_engineering.py - Technical indicators, feature merging, sequence creation
"""

import warnings
import numpy as np
import pandas as pd
from financial_indicators import build_advanced_indicators

warnings.filterwarnings("ignore")

try:
    import ta
    TA_AVAILABLE = True
except ImportError:
    TA_AVAILABLE = False

from util import log, build_sequences, clip_outliers


# ─────────────────────────────────────────────
# Technical Indicator Engine
# ─────────────────────────────────────────────
def calculate_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute comprehensive technical indicators.
    Accepts df with [Open, High, Low, Close, Volume].
    Returns enriched DataFrame.
    """
    df = df.copy()
    close  = df["Close"]
    high   = df["High"]
    low    = df["Low"]
    volume = df["Volume"]
    open_  = df["Open"]

    # ── Price-derived ─────────────────────────
    df["returns_1d"]  = close.pct_change(1)
    df["returns_5d"]  = close.pct_change(5)
    df["returns_20d"] = close.pct_change(20)
    df["log_return"]  = np.log(close / close.shift(1))
    df["hl_ratio"]    = (high - low) / (close + 1e-9)
    df["co_ratio"]    = (close - open_) / (open_ + 1e-9)

    # ── Moving averages ───────────────────────
    for w in [5, 10, 20, 50, 100, 200]:
        df[f"sma_{w}"]  = close.rolling(w).mean()
        df[f"ema_{w}"]  = close.ewm(span=w, adjust=False).mean()

    # Price relative to MAs
    for w in [20, 50, 100]:
        df[f"close_vs_sma{w}"] = close / (df[f"sma_{w}"] + 1e-9) - 1

    # ── Momentum ──────────────────────────────
    # RSI (14)
    delta = close.diff()
    gain  = delta.clip(lower=0).rolling(14).mean()
    loss  = (-delta.clip(upper=0)).rolling(14).mean()
    rs    = gain / (loss + 1e-9)
    df["rsi_14"] = 100 - 100 / (1 + rs)

    # RSI (6) for short-term
    gain6 = delta.clip(lower=0).rolling(6).mean()
    loss6 = (-delta.clip(upper=0)).rolling(6).mean()
    rs6   = gain6 / (loss6 + 1e-9)
    df["rsi_6"] = 100 - 100 / (1 + rs6)

    # Stochastic RSI
    rsi_min = df["rsi_14"].rolling(14).min()
    rsi_max = df["rsi_14"].rolling(14).max()
    df["stoch_rsi"] = (df["rsi_14"] - rsi_min) / (rsi_max - rsi_min + 1e-9)

    # MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    df["macd"]        = ema12 - ema26
    df["macd_signal"] = df["macd"].ewm(span=9, adjust=False).mean()
    df["macd_hist"]   = df["macd"] - df["macd_signal"]
    df["macd_norm"]   = df["macd"] / (close + 1e-9)

    # Momentum (10)
    df["momentum_10"] = close / (close.shift(10) + 1e-9) - 1
    df["momentum_20"] = close / (close.shift(20) + 1e-9) - 1
    df["momentum_60"] = close / (close.shift(60) + 1e-9) - 1

    # ── Trend ─────────────────────────────────
    # ADX (manual)
    df["tr"] = np.maximum(
        high - low,
        np.maximum(abs(high - close.shift(1)), abs(low - close.shift(1))),
    )
    df["dm_plus"]  = np.where((high - high.shift(1)) > (low.shift(1) - low),
                               np.maximum(high - high.shift(1), 0), 0)
    df["dm_minus"] = np.where((low.shift(1) - low) > (high - high.shift(1)),
                               np.maximum(low.shift(1) - low, 0), 0)
    atr14      = df["tr"].rolling(14).mean()
    dip        = df["dm_plus"].rolling(14).mean()  / (atr14 + 1e-9) * 100
    dim        = df["dm_minus"].rolling(14).mean() / (atr14 + 1e-9) * 100
    dx         = abs(dip - dim) / (dip + dim + 1e-9) * 100
    df["adx"]  = dx.rolling(14).mean()
    df["di_plus"]  = dip
    df["di_minus"] = dim
    df.drop(columns=["tr", "dm_plus", "dm_minus"], inplace=True)

    # ── Volatility ────────────────────────────
    # ATR
    high_low = high - low
    high_close_prev = abs(high - close.shift(1))
    low_close_prev  = abs(low  - close.shift(1))
    tr_vals = pd.concat([high_low, high_close_prev, low_close_prev], axis=1).max(axis=1)
    df["atr_14"] = tr_vals.rolling(14).mean()
    df["atr_norm"] = df["atr_14"] / (close + 1e-9)

    # Bollinger Bands
    sma20    = close.rolling(20).mean()
    std20    = close.rolling(20).std()
    bb_upper = sma20 + 2 * std20
    bb_lower = sma20 - 2 * std20
    df["bb_width"]  = (bb_upper - bb_lower) / (sma20 + 1e-9)
    df["bb_pct"]    = (close - bb_lower) / (bb_upper - bb_lower + 1e-9)

    # Historical volatility (20d, 60d)
    df["hv_20"] = df["log_return"].rolling(20).std() * np.sqrt(252)
    df["hv_60"] = df["log_return"].rolling(60).std() * np.sqrt(252)

    # ── Volume indicators ─────────────────────
    df["volume_sma20"] = volume.rolling(20).mean()
    df["volume_ratio"] = volume / (df["volume_sma20"] + 1e-9)
    df["volume_sma5"]  = volume.rolling(5).mean()

    # OBV
    obv = (np.sign(close.diff()) * volume).fillna(0).cumsum()
    df["obv"]         = obv
    df["obv_ema20"]   = obv.ewm(span=20, adjust=False).mean()
    df["obv_signal"]  = (obv - df["obv_ema20"]) / (df["obv_ema20"].abs() + 1e-9)

    # VWAP (rolling 20d approximation)
    typical = (high + low + close) / 3
    df["vwap_20"]     = (typical * volume).rolling(20).sum() / (volume.rolling(20).sum() + 1e-9)
    df["close_vs_vwap"] = close / (df["vwap_20"] + 1e-9) - 1

    # ── Candle patterns ───────────────────────
    df["doji"]     = (abs(close - open_) / (high - low + 1e-9) < 0.1).astype(float)
    df["bullish_engulf"] = (
        (close > open_) &
        (open_ < close.shift(1)) &
        (close > open_.shift(1))
    ).astype(float)

    # ── Calendar features ─────────────────────
    df["day_of_week"]   = df.index.dayofweek / 4.0
    df["month"]         = df.index.month / 12.0
    df["quarter"]       = df.index.quarter / 4.0

    # Cleanup
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    return df


# ─────────────────────────────────────────────
# Merge market context
# ─────────────────────────────────────────────
def merge_market_context(
    stock_df: pd.DataFrame,
    market_ctx: pd.DataFrame,
) -> pd.DataFrame:
    """Left-join market context features onto stock DataFrame."""
    if market_ctx.empty:
        return stock_df
    merged = stock_df.join(market_ctx, how="left")
    market_cols = market_ctx.columns.tolist()
    merged[market_cols] = merged[market_cols].ffill().bfill()
    return merged


# ─────────────────────────────────────────────
# Merge sentiment features
# ─────────────────────────────────────────────
def merge_sentiment_features(
    stock_df: pd.DataFrame,
    sentiment_df: pd.DataFrame,
) -> pd.DataFrame:
    """Join pre-computed sentiment features indexed by Date."""
    if sentiment_df is None or sentiment_df.empty:
        # Fill with neutral values
        stock_df["sentiment_score"]   = 0.0
        stock_df["positive_count"]    = 0
        stock_df["negative_count"]    = 0
        stock_df["news_volume"]       = 0
        return stock_df
    merged = stock_df.join(sentiment_df, how="left")
    sent_cols = ["sentiment_score", "positive_count", "negative_count", "news_volume"]
    for c in sent_cols:
        if c in merged.columns:
            merged[c] = merged[c].fillna(0)
    return merged


# ─────────────────────────────────────────────
# Feature column list (ordered)
# ─────────────────────────────────────────────
FEATURE_COLUMNS = [
    # Price
    "returns_1d", "returns_5d", "returns_20d", "log_return",
    "hl_ratio", "co_ratio",
    # MA relationships
    "close_vs_sma20", "close_vs_sma50", "close_vs_sma100",
    # Momentum
    "rsi_14", "rsi_6", "stoch_rsi",
    "macd_norm", "macd_hist",
    "momentum_10", "momentum_20", "momentum_60",
    # Trend
    "adx", "di_plus", "di_minus",
    # Volatility
    "atr_norm", "bb_width", "bb_pct", "hv_20", "hv_60",
    # Volume
    "volume_ratio", "obv_signal", "close_vs_vwap",
    # Patterns
    "doji", "bullish_engulf",
    # Calendar
    "day_of_week", "month", "quarter",
    # Sentiment (may be 0 if unavailable)
    "sentiment_score", "positive_count", "negative_count", "news_volume",
]

MARKET_CONTEXT_COLS = [
    "NSEI_ret1",  "NSEI_ret5",  "NSEI_ret20",
    "NSEI_above_sma", "NSEI_vol20", "NSEI_rsi",
    "BSESN_ret1", "BSESN_ret5", "BSESN_ret20",
    "BSESN_above_sma", "BSESN_vol20", "BSESN_rsi",
]


def get_feature_columns(df: pd.DataFrame) -> list:
    """Return only columns that exist in df."""
    available = [c for c in FEATURE_COLUMNS + MARKET_CONTEXT_COLS if c in df.columns]
    return available


# ─────────────────────────────────────────────
# Prepare final sequences
# ─────────────────────────────────────────────
def create_feature_sequences(
    df: pd.DataFrame,
    labels: pd.Series,
    seq_len: int = 60,
    feature_cols: list = None,
) -> tuple:
    """
    Align features with labels, handle NaN, normalize, build sequences.
    Returns (X, y, scaler_params) where X.shape = (N, seq_len, n_features).
    """
    if feature_cols is None:
        feature_cols = get_feature_columns(df)

    # Align
    combined = df[feature_cols].copy()
    combined["__label__"] = labels

    # Drop rows where label is NaN (future not available)
    combined.dropna(subset=["__label__"], inplace=True)
    # Forward-fill remaining feature NaN
    combined[feature_cols] = combined[feature_cols].ffill().bfill()
    combined.dropna(inplace=True)

    if len(combined) < seq_len + 30:
        log.warning("Not enough rows to build sequences after cleaning.")
        return np.array([]), np.array([]), {}

    features = combined[feature_cols].values
    labs     = combined["__label__"].values

    # Clip outliers per feature
    for i in range(features.shape[1]):
        col_data = pd.Series(features[:, i])
        features[:, i] = clip_outliers(col_data).values

    # Z-score normalization (compute per feature)
    scaler_params = {}
    for i, col in enumerate(feature_cols):
        mu    = features[:, i].mean()
        sigma = features[:, i].std() + 1e-9
        features[:, i] = (features[:, i] - mu) / sigma
        scaler_params[col] = {"mean": float(mu), "std": float(sigma)}

    # Clip label outliers (robust regression)
    labs = pd.Series(labs)
    labs = clip_outliers(labs, n_std=3.0).values

    X, y = build_sequences(features, labs, seq_len=seq_len)
    log.info(f"Sequences built: X={X.shape}, y={y.shape}, features={len(feature_cols)}")
    return X, y, scaler_params


def apply_scaler(df: pd.DataFrame, scaler_params: dict, feature_cols: list) -> np.ndarray:
    """Apply stored normalization parameters to inference data."""
    data = df[feature_cols].values.copy().astype(np.float32)
    for i, col in enumerate(feature_cols):
        if col in scaler_params:
            mu    = scaler_params[col]["mean"]
            sigma = scaler_params[col]["std"]
            data[:, i] = (data[:, i] - mu) / sigma
    return data



from financial_indicators import build_advanced_indicators   # new import at top

def generate_financial_indicators(
    ohlcv:         pd.DataFrame,
    index_close:   pd.Series = None,
    sector_closes: pd.DataFrame = None,
) -> pd.DataFrame:
    """
    Wrapper that builds all advanced indicators for one stock.
    Returns a DataFrame aligned to ohlcv.index.
    Safe to call with index_close=None — relative strength simply skipped.
    """
    return build_advanced_indicators(ohlcv, index_close, sector_closes)


def merge_indicator_features(
    feat_df:    pd.DataFrame,
    adv_df:     pd.DataFrame,
) -> pd.DataFrame:
    """
    Merge advanced indicators into existing feature DataFrame.
    Aligns on index — any date gaps filled forward then backward.
    """
    merged = feat_df.join(adv_df, how="left", rsuffix="_adv")
    merged = merged.ffill().bfill()
    log.info(f"Features after merge: {len(merged.columns)} columns")
    return merged


def normalize_features(
    feat_df:       pd.DataFrame,
    scaler_params: dict,
    feature_cols:  list,
) -> pd.DataFrame:
    """
    Apply saved z-score scaler to a feature DataFrame.
    Uses pre-fitted mu/sigma from training — never recomputes.
    Called at inference time after generate_financial_indicators().
    """
    out = feat_df[feature_cols].copy()
    out.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    out = out.ffill().bfill().fillna(0)
    for i, col in enumerate(feature_cols):
        if col in scaler_params:
            mu    = scaler_params[col]["mean"]
            sigma = scaler_params[col]["std"]
            out[col] = (out[col] - mu) / (sigma + 1e-9)
    return out