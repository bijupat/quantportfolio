"""
hybrid_main.py — Triple-Branch Hybrid Model (renamed from composite_main.py)
=============================================================================
Trains a deep learning model that fuses three information streams:

  Branch 1 — Time Series     : 120 days of OHLCV + technical indicators
  Branch 2 — Financial Data  : ROE, ROA, PE, revenue growth, D/E,
                                FCF, Piotroski F-Score, Altman Z-Score
  Branch 3 — News Sentiment  : 60-day rolling daily sentiment features

Multi-symbol training:
  Like main.py, this file accepts --symbols (multiple tickers).
  Feature matrices from ALL symbols are concatenated before training —
  the model learns cross-stock patterns, not just single-stock behaviour.
  The scaler is fit on the combined data and saved per symbol so
  predict.py can normalise each symbol correctly at inference time.

Design constraint (critical):
  The trained model MUST be loadable by the existing load_model() in
  predict.py / ensemble_predict.py / composite_predict.py WITHOUT any
  code changes.

  Those loaders call:
      model(x, training=False)   where x.shape = (batch, seq_len, n_features)

  Solution: financial and sentiment features are normalised and
  appended as EXTRA COLUMNS to the time-series feature matrix.
  The resulting tensor shape is (batch, seq_len, n_ts + n_fin + n_sent)
  which is just a wider model input — fully compatible.

Saved files (for existing loaders):
  models/<model_name>.weights.h5
  metrics/<model_name>_arch.json
  metrics/<model_name>_scalers.json

Also saved (versioned archive):
  models_storage/hybrid/hybrid_<model_name>_<DDMonYY>.weights.h5
  models_storage/hybrid/hybrid_<model_name>_<DDMonYY>_arch.json
  models_storage/hybrid/hybrid_<model_name>_<DDMonYY>_scalers.json

Usage:
  python hybrid_main.py
  python hybrid_main.py --symbols RELIANCE.NS TCS.NS INFY.NS
  python hybrid_main.py --symbols HDFCBANK.NS ICICIBANK.NS --model_name hybrid_banking
  python hybrid_main.py --symbols RELIANCE.NS TCS.NS --epochs 60 --no-sentiment
"""

import os
import sys
import json
import argparse
import warnings
import datetime
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

# ── Project imports ───────────────────────────────────────────
from util import log, MODELS_DIR, METRICS_DIR, save_json, clip_outliers
from data_loader import (
    fetch_stock_data, fetch_index_data, build_market_context,
)
from feature_engineering import (
    calculate_technical_indicators,
    merge_market_context, merge_sentiment_features,
    FEATURE_COLUMNS, MARKET_CONTEXT_COLS,
)
from news_sentiment import fetch_news_sentiment
from model_transformer import (
    QuantTransformer, build_transformer_model,
    evaluate_model as base_evaluate_model,
)
from portfolio_optimizer import walk_forward_splits

# ═══════════════════════════════════════════════════════════════
# CONFIGURATION
# ═══════════════════════════════════════════════════════════════

# Archive folder for versioned hybrid model files
HYBRID_STORAGE = Path("models_storage") / "hybrid"

# Financial feature names (fetched from yfinance .info)
FINANCIAL_FEATURES = [
    "returnOnEquity",        # ROE
    "returnOnAssets",        # ROA
    "trailingPE",            # P/E ratio
    "revenueGrowth",         # Revenue growth YoY
    "debtToEquity",          # Debt/Equity ratio
    "freeCashflow_norm",     # Free cash flow / market cap (normalised)
    "piotroski_score_norm",  # Piotroski F-Score / 9
    "altman_z_norm",         # min(Altman Z / 5, 1.0) — 0 for financial stocks
]

# Sentiment feature names (from news_sentiment.py daily output)
SENTIMENT_FEATURES = [
    "sentiment_score",
    "positive_count",
    "negative_count",
    "news_volume",
]

# Number of days for each data stream
TS_DAYS       = 180    # days of price history to fetch (120 used in sequences)
TS_SEQ_LEN    = 120    # transformer lookback window
SENT_DAYS     = 60     # rolling sentiment window appended per timestep
HORIZON       = 30     # prediction target: 30-day forward return


# ═══════════════════════════════════════════════════════════════
# SECTION 1 — FINANCIAL DATA LOADER
# Fetches fundamentals from yfinance and Piotroski/Altman modules.
# Returns a single flat dict of scalar values (one value per feature).
# Forward-filled to daily frequency in the alignment step.
# ═══════════════════════════════════════════════════════════════

def load_financial_data(symbol: str) -> Dict[str, float]:
    """
    Fetch fundamental financial data for one symbol.

    Returns a flat dict of scalar ratios:
        {feature_name: value}

    All values are normalised to a roughly 0-1 range so they can be
    appended as extra columns alongside z-scored technical features.

    Missing values default to 0.0 (neutral) rather than raising errors —
    the model is trained on these defaults and handles them gracefully.
    """
    import yfinance as yf

    result = {f: 0.0 for f in FINANCIAL_FEATURES}

    try:
        info = yf.Ticker(symbol).info or {}
    except Exception as e:
        log.warning(f"  [{symbol}] yfinance .info failed: {e} — using zero financials")
        return result

    def _get(key, scale=1.0, clip_lo=-10.0, clip_hi=10.0, default=0.0):
        """Safe extract with optional scaling and clipping."""
        v = info.get(key)
        if v is None or not np.isfinite(float(v)):
            return default
        return float(np.clip(v * scale, clip_lo, clip_hi))

    # ROE: typically -1 to +1.5 in practice → clip and keep as-is
    result["returnOnEquity"]  = _get("returnOnEquity",  clip_lo=-2.0, clip_hi=2.0)
    # ROA: typically -0.3 to +0.3
    result["returnOnAssets"]  = _get("returnOnAssets",  clip_lo=-1.0, clip_hi=1.0)
    # PE: normalise by dividing by 100 → 0 to ~1 for reasonable PEs
    result["trailingPE"]      = _get("trailingPE", scale=0.01, clip_lo=0.0, clip_hi=1.0)
    # Revenue growth: typically -0.5 to +0.5
    result["revenueGrowth"]   = _get("revenueGrowth",   clip_lo=-1.0, clip_hi=1.0)
    # D/E: scale to 0-1 range (D/E > 3 is very high leverage)
    result["debtToEquity"]    = _get("debtToEquity", scale=0.01, clip_lo=0.0, clip_hi=1.0)

    # Free cash flow normalised by market cap
    fcf    = info.get("freeCashflow")
    mktcap = info.get("marketCap")
    if fcf is not None and mktcap and mktcap > 0 and np.isfinite(float(fcf)):
        result["freeCashflow_norm"] = float(np.clip(fcf / mktcap, -0.5, 0.5))

    # Piotroski F-Score (0-9) → normalise to 0-1
    try:
        from piotroski import calculate_piotroski_score
        p = calculate_piotroski_score(symbol)
        score = p.get("piotroski_score")
        if score is not None:
            result["piotroski_score_norm"] = float(score) / 9.0
    except Exception as e:
        log.debug(f"  [{symbol}] Piotroski unavailable: {e}")

    # Altman Z-Score → normalise to 0-1 (cap at Z=5)
    try:
        from altman import calculate_altman_z_score, _is_financial_sector
        if not _is_financial_sector(symbol):
            a = calculate_altman_z_score(symbol)
            z = a.get("z_score")
            if z is not None:
                result["altman_z_norm"] = float(np.clip(z / 5.0, 0.0, 1.0))
    except Exception as e:
        log.debug(f"  [{symbol}] Altman unavailable: {e}")

    log.info(
        f"  [{symbol}] Financials: ROE={result['returnOnEquity']:.3f}  "
        f"ROA={result['returnOnAssets']:.3f}  PE={result['trailingPE']:.3f}  "
        f"Piotroski={result['piotroski_score_norm']:.2f}  "
        f"Altman={result['altman_z_norm']:.2f}"
    )
    return result


# ═══════════════════════════════════════════════════════════════
# SECTION 2 — DATA ALIGNMENT
# Aligns three streams to the same daily trading-day index.
# Financial scalars → broadcast to every row (quarterly data forward-filled).
# Sentiment → daily aggregation → rolling mean → aligned to price dates.
# ═══════════════════════════════════════════════════════════════

def align_all_features(
    ts_df:        pd.DataFrame,
    financial:    Dict[str, float],
    sentiment_df: pd.DataFrame,
    sent_roll:    int = 5,
) -> pd.DataFrame:
    """
    Merge time-series, financial, and sentiment features into one DataFrame
    on the same daily date index as ts_df.

    Rules applied:
      Financial:  scalar repeated on every row (quarterly data is stable
                  across daily windows — forward-fill is implicit).
      Sentiment:  joined on date index, missing dates forward-filled,
                  then rolling mean applied to reduce daily noise.
      All:        remaining NaNs filled with 0.0.

    Args:
        ts_df:        DataFrame from calculate_technical_indicators() —
                      rows are trading days, columns are TS features
        financial:    flat dict of scalar fundamental ratios
        sentiment_df: daily sentiment DataFrame from fetch_news_sentiment()
        sent_roll:    rolling window (days) to smooth sentiment noise

    Returns:
        Wide DataFrame with columns: [TS features | financial | sentiment]
        Indexed by trading date, same length as ts_df.
    """
    out = ts_df.copy()

    # ── Financial: broadcast scalar to every row ──────────────
    for feat, val in financial.items():
        out[feat] = val   # same value on every day — fundamental data is static

    # ── Sentiment: join + forward-fill + rolling smooth ───────
    if sentiment_df is not None and not sentiment_df.empty:
        sent = sentiment_df.reindex(out.index).ffill().bfill().fillna(0)

        # Apply rolling mean to reduce event-driven noise (Devil's Advocate pt 2)
        for col in SENTIMENT_FEATURES:
            if col in sent.columns:
                sent[col] = sent[col].rolling(sent_roll, min_periods=1).mean()
        out = out.join(sent[SENTIMENT_FEATURES], how="left")
    else:
        # No sentiment data — pad with zeros (neutral)
        for col in SENTIMENT_FEATURES:
            out[col] = 0.0

    out = out.ffill().bfill().fillna(0.0)
    return out


# ═══════════════════════════════════════════════════════════════
# SECTION 3 — TRIPLE-BRANCH ARCHITECTURE
#
# Internal architecture (training only):
#   Branch 1  (seq_len, n_ts)       → LSTM → Dense → embedding_ts
#   Branch 2  (n_fin,)              → Dense → Dense → embedding_fin
#   Branch 3  (seq_len, n_sent)     → TransformerEncoder → Pool → embedding_sent
#   Fusion    concat(emb_ts, emb_fin, emb_sent) → Dense → output
#
# External interface (compatible with predict.py):
#   input shape: (batch, seq_len, n_ts + n_fin + n_sent)
#   The call() method slices the tensor internally into three branches.
#   Output: scalar 30-day return score  (same as QuantTransformer)
#
# This design means load_model() in all existing files works unchanged —
# they pass a (batch, seq_len, n_features) tensor and get a scalar back.
# ═══════════════════════════════════════════════════════════════

class TripleBranchHybridModel(keras.Model):
    """
    Triple-branch deep learning model for stock return prediction.

    Accepts a SINGLE wide tensor (batch, seq_len, n_total_features) where
    n_total_features = n_ts_features + n_financial_features + n_sentiment_features.
    Internally slices into three branches, processes each with a specialised
    sub-network, fuses the embeddings, and regresses to a scalar return.

    This single-tensor interface makes the model drop-in compatible with
    predict.py, ensemble_predict.py, and composite_predict.py without
    any changes to those files.

    Attributes stored for load_model() compatibility:
        seq_len, n_features, d_model, n_heads, n_layers
        n_ts_features, n_financial_features, n_sentiment_features
    """

    def __init__(
        self,
        seq_len:              int   = 120,
        n_ts_features:        int   = 50,   # time-series feature count
        n_financial_features: int   = 8,    # len(FINANCIAL_FEATURES)
        n_sentiment_features: int   = 4,    # len(SENTIMENT_FEATURES)
        d_model:              int   = 64,
        n_heads:              int   = 4,
        n_layers:             int   = 2,
        lstm_units:           int   = 64,
        dropout:              float = 0.15,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.seq_len              = seq_len
        self.n_ts_features        = n_ts_features
        self.n_financial_features = n_financial_features
        self.n_sentiment_features = n_sentiment_features
        self.n_features           = n_ts_features + n_financial_features + n_sentiment_features
        self.d_model              = d_model
        self.n_heads              = n_heads
        self.n_layers             = n_layers
        self.lstm_units           = lstm_units

        # ── Branch 1: Time-Series (LSTM) ──────────────────────
        # LSTM captures sequential momentum patterns
        self.ts_lstm   = layers.LSTM(lstm_units, return_sequences=False, dropout=dropout)
        self.ts_norm   = layers.LayerNormalization(epsilon=1e-6)
        self.ts_dense  = layers.Dense(d_model, activation="gelu")
        self.ts_drop   = layers.Dropout(dropout)

        # ── Branch 2: Financial (Dense MLP) ───────────────────
        # Financial ratios are static scalars — no sequence needed
        self.fin_dense1 = layers.Dense(32, activation="gelu")
        self.fin_norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.fin_drop1  = layers.Dropout(dropout)
        self.fin_dense2 = layers.Dense(d_model, activation="gelu")
        self.fin_drop2  = layers.Dropout(dropout)

        # ── Branch 3: Sentiment (Transformer Encoder) ─────────
        # Transformer self-attention over the sentiment time-series
        self.sent_proj   = layers.Dense(d_model, use_bias=False)
        self.sent_norm_i = layers.LayerNormalization(epsilon=1e-6)

        # One transformer encoder block for sentiment
        self.sent_attn   = layers.MultiHeadAttention(
            num_heads = max(1, n_heads // 2),
            key_dim   = max(1, d_model // max(1, n_heads // 2)),
        )
        self.sent_ffn    = keras.Sequential([
            layers.Dense(d_model * 2, activation="gelu"),
            layers.Dropout(dropout),
            layers.Dense(d_model),
        ])
        self.sent_norm1  = layers.LayerNormalization(epsilon=1e-6)
        self.sent_norm2  = layers.LayerNormalization(epsilon=1e-6)
        self.sent_drop   = layers.Dropout(dropout)

        # Attention pooling over sentiment sequence
        self.sent_pool_q = layers.Dense(1)
        self.sent_drop2  = layers.Dropout(dropout)

        # ── Fusion layer ───────────────────────────────────────
        # Concatenates three d_model embeddings → 3 * d_model
        self.fusion_dense1 = layers.Dense(d_model * 2, activation="gelu")
        self.fusion_norm   = layers.LayerNormalization(epsilon=1e-6)
        self.fusion_drop   = layers.Dropout(dropout)
        self.fusion_dense2 = layers.Dense(d_model, activation="gelu")

        # ── Regression head ────────────────────────────────────
        self.head = keras.Sequential([
            layers.Dense(64, activation="gelu"),
            layers.Dropout(dropout * 0.5),
            layers.Dense(1, activation="linear"),
        ])

        # Compatibility attributes expected by load_model() callers
        self._last_attn_weights = None

    def call(self, x, training=False):
        """
        Forward pass.

        Args:
            x: tensor shape (batch, seq_len, n_total_features)
               Columns are ordered: [ts_features | financial | sentiment]
               This ordering is enforced by build_feature_matrix().

        Returns:
            scalar predicted 30-day return, shape (batch,)
        """
        # ── Slice input into three branches ───────────────────
        n_ts   = self.n_ts_features
        n_fin  = self.n_financial_features
        n_sent = self.n_sentiment_features

        x_ts   = x[:, :, :n_ts]                           # (B, T, n_ts)
        x_fin  = x[:, 0, n_ts:n_ts + n_fin]               # (B, n_fin)  — take row 0 (static)
        x_sent = x[:, :, n_ts + n_fin:]                   # (B, T, n_sent)

        # ── Branch 1: Time-Series ──────────────────────────────
        ts_out = self.ts_lstm(x_ts, training=training)    # (B, lstm_units)
        ts_out = self.ts_norm(ts_out)
        ts_emb = self.ts_drop(self.ts_dense(ts_out), training=training)  # (B, d_model)

        # ── Branch 2: Financial ────────────────────────────────
        fin_out = self.fin_dense1(x_fin)                  # (B, 32)
        fin_out = self.fin_norm1(fin_out)
        fin_out = self.fin_drop1(fin_out, training=training)
        fin_emb = self.fin_drop2(self.fin_dense2(fin_out), training=training)  # (B, d_model)

        # ── Branch 3: Sentiment ────────────────────────────────
        s = self.sent_proj(x_sent)                        # (B, T, d_model)
        s = self.sent_norm_i(s)

        # Transformer self-attention block
        s_norm  = self.sent_norm1(s)
        s_attn, attn_w = self.sent_attn(
            s_norm, s_norm, return_attention_scores=True, training=training
        )
        s = s + self.sent_drop(s_attn, training=training)
        s_norm = self.sent_norm2(s)
        s = s + self.sent_ffn(s_norm, training=training)

        self._last_attn_weights = [attn_w]   # store for compatibility

        # Attention pooling over sentiment sequence
        scores  = self.sent_pool_q(s)                     # (B, T, 1)
        weights = tf.nn.softmax(scores, axis=1)           # (B, T, 1)
        sent_emb = tf.reduce_sum(s * weights, axis=1)     # (B, d_model)
        sent_emb = self.sent_drop2(sent_emb, training=training)

        # ── Fusion ─────────────────────────────────────────────
        fused = tf.concat([ts_emb, fin_emb, sent_emb], axis=-1)  # (B, 3*d_model)
        fused = self.fusion_norm(self.fusion_dense1(fused))
        fused = self.fusion_drop(fused, training=training)
        fused = self.fusion_dense2(fused)                 # (B, d_model)

        out = self.head(fused, training=training)         # (B, 1)
        return tf.squeeze(out, axis=-1)                   # (B,)


def build_hybrid_model(
    seq_len:              int   = 120,
    n_ts_features:        int   = 50,
    n_financial_features: int   = 8,
    n_sentiment_features: int   = 4,
    d_model:              int   = 64,
    n_heads:              int   = 4,
    n_layers:             int   = 2,
    lstm_units:           int   = 64,
    dropout:              float = 0.15,
    lr:                   float = 1e-4,
) -> TripleBranchHybridModel:
    """Build, compile, and warm-up the triple-branch model."""
    model = TripleBranchHybridModel(
        seq_len              = seq_len,
        n_ts_features        = n_ts_features,
        n_financial_features = n_financial_features,
        n_sentiment_features = n_sentiment_features,
        d_model              = d_model,
        n_heads              = n_heads,
        n_layers             = n_layers,
        lstm_units           = lstm_units,
        dropout              = dropout,
    )
    model.compile(
        optimizer = keras.optimizers.Adam(learning_rate=lr, clipnorm=1.0),
        loss      = "huber",
        metrics   = ["mae"],
    )
    n_total = n_ts_features + n_financial_features + n_sentiment_features
    dummy   = tf.zeros((1, seq_len, n_total))
    model(dummy, training=False)
    log.info(
        f"TripleBranchHybridModel built: "
        f"ts={n_ts_features}  fin={n_financial_features}  sent={n_sentiment_features}  "
        f"total_features={n_total}  params={model.count_params():,}"
    )
    return model


# ═══════════════════════════════════════════════════════════════
# SECTION 4 — FEATURE MATRIX BUILDER
# Produces a single wide DataFrame in the correct column order:
#   [ts_features | financial_features | sentiment_features]
# This ordering is sliced back apart in TripleBranchHybridModel.call()
# ═══════════════════════════════════════════════════════════════

def build_feature_matrix(
    symbol:       str,
    start:        str,
    end:          Optional[str],
    market_ctx:   pd.DataFrame,
    use_sentiment: bool = True,
) -> Tuple[pd.DataFrame, Dict[str, float], List[str]]:
    """
    Build the complete wide feature matrix for one symbol.

    Returns:
        feat_df      — DataFrame with columns ordered [ts | fin | sent]
        financial    — raw financial dict (for logging / metadata)
        feature_cols — ordered list of all column names (saved as scaler keys)
    """
    # ── Time-series features ──────────────────────────────────
    raw = fetch_stock_data(symbol, start=start, end=end)
    if raw is None or raw.empty:
        raise ValueError(f"No price data for {symbol}")

    ts_df = calculate_technical_indicators(raw)
    ts_df = merge_market_context(ts_df, market_ctx)

    # ── Sentiment features ────────────────────────────────────
    if use_sentiment:
        try:
            sent_df = fetch_news_sentiment(symbol, start=start, end=end)
        except Exception as e:
            log.warning(f"  [{symbol}] Sentiment failed: {e} — using zeros")
            sent_df = None
    else:
        sent_df = None

    # ── Financial features ────────────────────────────────────
    financial = load_financial_data(symbol)

    # ── Align all on trading-day index ────────────────────────
    feat_df = align_all_features(ts_df, financial, sent_df)

    # ── Determine ordered column list ─────────────────────────
    # Order: ts columns, then financial, then sentiment
    ts_cols   = [c for c in feat_df.columns if c in ts_df.columns]
    fin_cols  = [c for c in FINANCIAL_FEATURES if c in feat_df.columns]
    sent_cols = [c for c in SENTIMENT_FEATURES if c in feat_df.columns]
    feature_cols = ts_cols + fin_cols + sent_cols

    # Only keep defined columns, fill missing
    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0

    feat_df = feat_df[feature_cols].copy()
    feat_df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    feat_df = feat_df.ffill().bfill().fillna(0.0)

    return feat_df, financial, feature_cols


# ═══════════════════════════════════════════════════════════════
# SECTION 5 — NORMALISATION
# Z-score per feature, computed on training data.
# Identical contract to feature_engineering.py — saves to scaler JSON
# in the same format predict.py expects.
# ═══════════════════════════════════════════════════════════════

def fit_and_apply_scaler(
    features: np.ndarray,
    feature_cols: List[str],
    symbol: str,
) -> Tuple[np.ndarray, Dict]:
    """
    Z-score normalise features and return scaler params.
    Matches the scaler format saved by feature_engineering.py so
    predict.py can load and apply it at inference time.

    Returns:
        normalised  — float32 array, same shape as features
        scaler_dict — {col: {"mean": float, "std": float}}
    """
    scaler_dict = {}
    normalised  = features.copy().astype(np.float64)
    for i, col in enumerate(feature_cols):
        col_data   = features[:, i]
        # Clip outliers before computing stats (Devil's Advocate pt 3)
        col_series = clip_outliers(pd.Series(col_data))
        mu    = float(col_series.mean())
        sigma = float(col_series.std()) + 1e-9
        normalised[:, i] = (col_data - mu) / sigma
        scaler_dict[col] = {"mean": mu, "std": sigma}
    return normalised.astype(np.float32), scaler_dict


# ═══════════════════════════════════════════════════════════════
# SECTION 6 — LABEL CREATION
# 30-day forward return — identical to feature_engineering.py
# ═══════════════════════════════════════════════════════════════

def create_labels(close: pd.Series, horizon: int = 30) -> pd.Series:
    """future_return = (close[t+horizon] - close[t]) / close[t]"""
    future = close.shift(-horizon)
    labels = (future - close) / (close + 1e-9)
    labels.name = "future_return"
    return labels


# ═══════════════════════════════════════════════════════════════
# SECTION 7 — SEQUENCE BUILDER
# Identical sliding-window logic as util.build_sequences()
# ═══════════════════════════════════════════════════════════════

def build_sequences(
    features: np.ndarray,
    labels:   np.ndarray,
    seq_len:  int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Slide a seq_len window over features+labels."""
    X, y = [], []
    for i in range(len(features) - seq_len):
        X.append(features[i: i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ═══════════════════════════════════════════════════════════════
# SECTION 8 — MODEL SAVING
# Saves to TWO locations:
#   1. models/<name>.weights.h5 + metrics/<name>_arch.json + _scalers.json
#      → picked up by predict.py / ensemble_predict.py / composite_predict.py
#   2. models_storage/hybrid/hybrid_<name>_<date>.weights.h5
#      → versioned archive with Windows-style auto-numbering if name exists
# ═══════════════════════════════════════════════════════════════

def _versioned_path(base_path: Path) -> Path:
    """
    Return base_path if it doesn't exist, otherwise
    base_path(1), base_path(2), ... — Windows-style numbering.
    """
    if not base_path.exists():
        return base_path
    stem    = base_path.stem
    suffix  = base_path.suffix
    parent  = base_path.parent
    counter = 1
    while True:
        candidate = parent / f"{stem}({counter}){suffix}"
        if not candidate.exists():
            return candidate
        counter += 1


def save_hybrid_model(
    model:        TripleBranchHybridModel,
    model_name:   str,
    scaler_params: Dict,
    feature_cols:  List[str],
    n_ts_features: int,
    n_financial_features: int,
    n_sentiment_features: int,
) -> Tuple[Path, Path]:
    """
    Save model weights and metadata to both storage locations.

    Returns:
        (primary_path, archive_path)
        primary_path  — models/<name>.weights.h5  (for existing loaders)
        archive_path  — models_storage/hybrid/hybrid_<name>_<date>.weights.h5
    """
    import h5py

    arch = {
        "seq_len":              model.seq_len,
        "n_features":           model.n_features,
        "d_model":              model.d_model,
        "n_heads":              model.n_heads,
        "n_layers":             model.n_layers,
        "n_ts_features":        n_ts_features,
        "n_financial_features": n_financial_features,
        "n_sentiment_features": n_sentiment_features,
        "model_type":           "triple_branch_hybrid",
    }

    # ── Primary save: models/ + metrics/ ─────────────────────
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    primary_weights = MODELS_DIR / f"{model_name}.weights.h5"

    # Save weights as flat ordered arrays (h5py format — shape-matched loader)
    with h5py.File(primary_weights, "w") as f:
        for i, w in enumerate(model.weights):
            f.create_dataset(f"weight_{i:04d}", data=np.array(w))
        # Embed arch attrs for self-describing load
        for k, v in arch.items():
            if isinstance(v, (int, float, str)):
                f.attrs[k] = v

    # arch JSON
    save_json(arch, METRICS_DIR / f"{model_name}_arch.json")

    # Scalers JSON — {symbol: {col: {mean, std}}} format
    # Wrapped in a symbol key so predict.py's scaler loading works
    save_json(scaler_params, METRICS_DIR / f"{model_name}_scalers.json")

    log.info(f"  Primary save: {primary_weights}")

    # ── Archive save: models_storage/hybrid/ ─────────────────
    HYBRID_STORAGE.mkdir(parents=True, exist_ok=True)
    date_str   = datetime.datetime.now().strftime("%d%b%y")
    base_stem  = f"hybrid_{model_name}_{date_str}"
    arch_name  = _versioned_path(HYBRID_STORAGE / f"{base_stem}_arch.json")
    weights_name = _versioned_path(HYBRID_STORAGE / f"{base_stem}.weights.h5")
    scalers_name = _versioned_path(HYBRID_STORAGE / f"{base_stem}_scalers.json")

    import shutil
    shutil.copy2(primary_weights, weights_name)
    shutil.copy2(METRICS_DIR / f"{model_name}_arch.json", arch_name)
    shutil.copy2(METRICS_DIR / f"{model_name}_scalers.json", scalers_name)

    log.info(f"  Archive save: {weights_name.name}")
    return primary_weights, weights_name


# ═══════════════════════════════════════════════════════════════
# SECTION 9 — TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def train_hybrid_model(
    model:       TripleBranchHybridModel,
    X_train:     np.ndarray,
    y_train:     np.ndarray,
    X_val:       np.ndarray,
    y_val:       np.ndarray,
    model_name:  str,
    epochs:      int   = 60,
    batch_size:  int   = 32,
    patience:    int   = 12,
) -> keras.callbacks.History:
    """Train with early stopping and LR reduction."""

    tmp_path = MODELS_DIR / f"{model_name}_tmp_best.weights.h5"

    class _LogCB(keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            if (epoch + 1) % 5 == 0:
                logs = logs or {}
                log.info(
                    f"  Epoch {epoch+1:3d}  "
                    f"loss={logs.get('loss', 0):.5f}  "
                    f"mae={logs.get('mae', 0):.5f}  "
                    f"val_loss={logs.get('val_loss', 0):.5f}  "
                    f"val_mae={logs.get('val_mae', 0):.5f}"
                )

    callbacks = [
        keras.callbacks.EarlyStopping(
            monitor="val_mae", patience=patience,
            restore_best_weights=True, verbose=1,
        ),
        keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss", factor=0.5, patience=6,
            min_lr=1e-6, verbose=1,
        ),
        keras.callbacks.ModelCheckpoint(
            str(tmp_path), monitor="val_mae",
            save_best_only=True, save_weights_only=True, verbose=0,
        ),
        keras.callbacks.TerminateOnNaN(),
        _LogCB(),
    ]

    log.info(
        f"Training '{model_name}'  "
        f"Train={X_train.shape[0]:,}  Val={X_val.shape[0]:,}  "
        f"Features={X_train.shape[2]}  Epochs={epochs}  BS={batch_size}"
    )

    history = model.fit(
        X_train, y_train,
        validation_data = (X_val, y_val),
        epochs          = epochs,
        batch_size      = batch_size,
        callbacks       = callbacks,
        verbose         = 0,
    )

    # Cleanup tmp checkpoint
    if tmp_path.exists():
        tmp_path.unlink()

    best_val = min(history.history.get("val_mae", [999]))
    log.info(f"Training complete.  Best val_mae = {best_val:.5f}")
    return history


# ═══════════════════════════════════════════════════════════════
# SECTION 10 — PREDICTION FUNCTION
# Mirrors predict_symbol() from predict.py for single-symbol use.
# ═══════════════════════════════════════════════════════════════

def predict_next_30_days(
    symbol:        str,
    model:         TripleBranchHybridModel,
    scaler_params: Dict,          # {col: {mean, std}}
    feature_cols:  List[str],
    market_ctx:    pd.DataFrame,
    start:         str  = "2020-01-01",
    end:           Optional[str] = None,
    use_sentiment: bool = True,
) -> Tuple[float, float]:
    """
    Predict 30-day forward return and compute a simple confidence score.

    Args:
        symbol:        e.g. 'RELIANCE.NS'
        model:         trained TripleBranchHybridModel
        scaler_params: {col: {mean, std}} loaded from _scalers.json
        feature_cols:  ordered list of feature names (must match training)
        market_ctx:    from build_market_context()
        start/end:     date range for price history fetch
        use_sentiment: whether to fetch news sentiment

    Returns:
        (predicted_return, confidence_score)
        predicted_return: float (e.g. 0.068 = +6.8%)
        confidence_score: float 0-1 (based on model output magnitude + data quality)
    """
    feat_df, financial, _ = build_feature_matrix(
        symbol, start, end, market_ctx, use_sentiment
    )

    # Align to expected feature_cols
    for c in feature_cols:
        if c not in feat_df.columns:
            feat_df[c] = 0.0
    feat_df = feat_df[feature_cols].copy()
    feat_df.replace([float("inf"), float("-inf")], float("nan"), inplace=True)
    feat_df = feat_df.ffill().bfill().fillna(0.0)

    if len(feat_df) < model.seq_len:
        log.warning(f"  [{symbol}] Insufficient rows ({len(feat_df)}) for seq_len={model.seq_len}")
        return float("nan"), 0.0

    # Apply saved scaler
    window = feat_df.iloc[-model.seq_len:].values.astype("float32")
    for i, col in enumerate(feature_cols):
        if col in scaler_params:
            mu    = scaler_params[col]["mean"]
            sigma = scaler_params[col]["std"]
            window[:, i] = (window[:, i] - mu) / (sigma + 1e-9)

    x    = tf.constant(window[np.newaxis], dtype=tf.float32)
    pred = float(model(x, training=False).numpy().flatten()[0])

    # Confidence: sigmoid of absolute prediction magnitude (heuristic)
    # Strong signal (|pred| > 0.1) → confidence close to 1
    # Weak signal  (|pred| ≈ 0.0) → confidence ≈ 0.5
    confidence = float(1 / (1 + np.exp(-abs(pred) * 20)))

    return pred, confidence


# ═══════════════════════════════════════════════════════════════
# SECTION 11 — DRIVER FUNCTIONS
# ═══════════════════════════════════════════════════════════════

def load_time_series_data(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Wrapper: fetch OHLCV for one symbol."""
    return fetch_stock_data(symbol, start=start, end=end) or pd.DataFrame()


def load_sentiment_data(symbol: str, start: str, end: Optional[str]) -> pd.DataFrame:
    """Wrapper: fetch daily sentiment for one symbol."""
    try:
        return fetch_news_sentiment(symbol, start=start, end=end)
    except Exception as e:
        log.warning(f"  [{symbol}] Sentiment load failed: {e}")
        return pd.DataFrame()


def prepare_time_series_features(raw: pd.DataFrame, market_ctx: pd.DataFrame) -> pd.DataFrame:
    """Apply technical indicators + market context."""
    return merge_market_context(calculate_technical_indicators(raw), market_ctx)


def prepare_financial_features(symbol: str) -> Dict[str, float]:
    """Alias for load_financial_data — named per prompt spec."""
    return load_financial_data(symbol)


def prepare_sentiment_features(sent_df: pd.DataFrame, roll: int = 5) -> pd.DataFrame:
    """Apply rolling smoothing to daily sentiment (noise reduction)."""
    if sent_df is None or sent_df.empty:
        return pd.DataFrame()
    smoothed = sent_df.copy()
    for col in SENTIMENT_FEATURES:
        if col in smoothed.columns:
            smoothed[col] = smoothed[col].rolling(roll, min_periods=1).mean()
    return smoothed


def normalize_all_features(
    features:     np.ndarray,
    feature_cols: List[str],
    symbol:       str,
) -> Tuple[np.ndarray, Dict]:
    """Alias for fit_and_apply_scaler — named per prompt spec."""
    return fit_and_apply_scaler(features, feature_cols, symbol)


# ═══════════════════════════════════════════════════════════════
# SECTION 12 — MAIN TRAINING PIPELINE
# ═══════════════════════════════════════════════════════════════

def main():
    parser = argparse.ArgumentParser(
        description="QuantPortfolioAI — Triple-Branch Hybrid Model Training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python hybrid_main.py
  python hybrid_main.py --symbols RELIANCE.NS TCS.NS INFY.NS
  python hybrid_main.py --symbols HDFCBANK.NS ICICIBANK.NS --model_name hybrid_banking
  python hybrid_main.py --symbols RELIANCE.NS TCS.NS --epochs 60 --no-sentiment
        """,
    )
    # ── Multi-symbol: nargs="+" accepts one or more tickers ───────────────────
    parser.add_argument(
        "--symbols", nargs="+",
        default=["RELIANCE.NS", "TCS.NS", "HDFCBANK.NS",
                 "INFY.NS", "ICICIBANK.NS"],
        help="Stock symbols to train on (default: 5 NIFTY50 stocks)",
    )
    parser.add_argument("--model_name",   default="default_model",
                        help="Model save name (default: default_model)")
    parser.add_argument("--start",        default="2015-01-01")
    parser.add_argument("--end",          default=None)
    parser.add_argument("--epochs",       type=int, default=60)
    parser.add_argument("--batch-size",   type=int, default=32)
    parser.add_argument("--d-model",      type=int, default=64)
    parser.add_argument("--n-heads",      type=int, default=4)
    parser.add_argument("--n-layers",     type=int, default=2)
    parser.add_argument("--lstm-units",   type=int, default=64)
    parser.add_argument("--seq-len",      type=int, default=120)
    parser.add_argument("--no-sentiment", action="store_true",
                        help="Skip news sentiment branch")
    args = parser.parse_args()

    log.info("=" * 60)
    log.info("  QuantPortfolioAI — Triple-Branch Hybrid Model")
    log.info("=" * 60)
    log.info(f"  Symbols    : {args.symbols}")
    log.info(f"  Model name : {args.model_name}")
    log.info(f"  Sentiment  : {'disabled' if args.no_sentiment else 'enabled'}")

    # ── Market context (shared for all symbols) ───────────────────────────────
    log.info("\nStep 1: Fetching market context…")
    index_raw  = fetch_index_data(start=args.start, end=args.end)
    market_ctx = build_market_context(index_raw)

    # ── Step 2-4: Build feature matrix for each symbol, then concatenate ─────
    # This mirrors main.py's multi-stock approach:
    #   sequences from all symbols are pooled before training so the model
    #   learns cross-stock patterns, not just single-stock behaviour.
    log.info("\nStep 2-4: Building feature matrices for all symbols…")

    all_X:           List[np.ndarray] = []
    all_y:           List[np.ndarray] = []
    all_scalers:     Dict             = {}   # {symbol: {col: {mean, std}}}
    feature_cols:    List[str]        = []   # set from first valid symbol
    last_financial:  Dict             = {}   # for demo prediction output
    last_symbol:     str              = args.symbols[0]

    for sym in args.symbols:
        log.info(f"\n  Processing: {sym}")
        try:
            feat_df, financial, fcols = build_feature_matrix(
                symbol        = sym,
                start         = args.start,
                end           = args.end,
                market_ctx    = market_ctx,
                use_sentiment = not args.no_sentiment,
            )
        except Exception as e:
            log.warning(f"  [{sym}] Feature matrix failed: {e} — skipping")
            continue

        # Use first valid symbol to set the column list for the whole model
        if not feature_cols:
            feature_cols = fcols
        else:
            # Align to established column list — fill missing with 0
            for c in feature_cols:
                if c not in feat_df.columns:
                    feat_df[c] = 0.0
            feat_df = feat_df[feature_cols]

        # Labels
        raw_sym = fetch_stock_data(sym, start=args.start, end=args.end)
        if raw_sym is None or raw_sym.empty:
            log.warning(f"  [{sym}] No price data — skipping")
            continue
        labels = create_labels(raw_sym["Close"], horizon=HORIZON)

        combined = feat_df.copy()
        combined["__label__"] = labels.reindex(feat_df.index)
        combined.dropna(subset=["__label__"], inplace=True)
        combined = combined.ffill().bfill().fillna(0.0)

        if len(combined) < args.seq_len + 50:
            log.warning(f"  [{sym}] Only {len(combined)} rows after cleanup — skipping")
            continue

        feat_arr  = combined[feature_cols].values.astype(np.float32)
        label_arr = combined["__label__"].values.astype(np.float32)

        # Fit scaler on this symbol's data
        feat_norm, scaler_dict = fit_and_apply_scaler(feat_arr, feature_cols, sym)
        all_scalers[sym] = scaler_dict

        # Build sequences
        X_sym, y_sym = build_sequences(feat_norm, label_arr, seq_len=args.seq_len)
        log.info(f"  [{sym}] sequences={len(X_sym):,}  features={feat_norm.shape[1]}")

        all_X.append(X_sym)
        all_y.append(y_sym)
        last_financial = financial
        last_symbol    = sym

    if not all_X:
        log.error("No valid symbols produced sequences. Check data availability.")
        sys.exit(1)

    # Concatenate all symbols then shuffle (cross-stock generalisation)
    X = np.concatenate(all_X, axis=0)
    y = np.concatenate(all_y, axis=0)
    shuffle_idx = np.random.permutation(len(X))
    X, y = X[shuffle_idx], y[shuffle_idx]
    log.info(
        f"\n  Combined: {X.shape[0]:,} sequences from {len(all_X)} symbols  "
        f"features={X.shape[2]}"
    )

    if len(X) < 100:
        log.error(f"Insufficient total sequences ({len(X)}). Add more symbols or use --start 2010-01-01")
        sys.exit(1)

    # ── Step 5: Train/Val/Test split ──────────────────────────────────────────
    log.info("\nStep 5: Splitting train/val/test…")
    splits = walk_forward_splits(len(X))
    train_sl, val_sl, test_sl = splits[0]
    X_train, y_train = X[train_sl], y[train_sl]
    X_val,   y_val   = X[val_sl],   y[val_sl]
    X_test,  y_test  = X[test_sl],  y[test_sl]
    log.info(f"  Train={len(X_train):,}  Val={len(X_val):,}  Test={len(X_test):,}")

    # ── Step 6: Build model ───────────────────────────────────────────────────
    log.info("\nStep 6: Building triple-branch model…")
    n_ts_features   = len([c for c in feature_cols
                           if c not in FINANCIAL_FEATURES + SENTIMENT_FEATURES])
    n_fin_features  = len([c for c in feature_cols if c in FINANCIAL_FEATURES])
    n_sent_features = len([c for c in feature_cols if c in SENTIMENT_FEATURES])

    model = build_hybrid_model(
        seq_len              = args.seq_len,
        n_ts_features        = n_ts_features,
        n_financial_features = n_fin_features,
        n_sentiment_features = n_sent_features,
        d_model              = args.d_model,
        n_heads              = args.n_heads,
        n_layers             = args.n_layers,
        lstm_units           = args.lstm_units,
    )

    # ── Step 7: Train ─────────────────────────────────────────────────────────
    log.info("\nStep 7: Training…")
    history = train_hybrid_model(
        model      = model,
        X_train    = X_train,
        y_train    = y_train,
        X_val      = X_val,
        y_val      = y_val,
        model_name = args.model_name,
        epochs     = args.epochs,
        batch_size = args.batch_size,
    )

    # ── Step 8: Evaluate ──────────────────────────────────────────────────────
    log.info("\nStep 8: Evaluating on test set…")
    y_pred = model.predict(X_test, verbose=0).flatten()
    from scipy.stats import spearmanr
    mse  = float(np.mean((y_test - y_pred) ** 2))
    mae  = float(np.mean(np.abs(y_test - y_pred)))
    hit  = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic,_ = spearmanr(y_test, y_pred)
    eval_metrics = {
        "MSE": mse, "MAE": mae,
        "HitRatio": hit, "IC_Spearman": float(ic),
        "symbols_trained": args.symbols,
        "n_sequences": int(len(X)),
    }
    save_json(eval_metrics, METRICS_DIR / f"{args.model_name}_eval.json")
    log.info(f"  Test  MAE={mae:.5f}  Hit={hit:.2%}  IC={float(ic):.4f}")

    # ── Step 9: Save ──────────────────────────────────────────────────────────
    log.info("\nStep 9: Saving model…")
    primary, archive = save_hybrid_model(
        model                = model,
        model_name           = args.model_name,
        scaler_params        = all_scalers,   # all symbols' scalers
        feature_cols         = feature_cols,
        n_ts_features        = n_ts_features,
        n_financial_features = n_fin_features,
        n_sentiment_features = n_sent_features,
    )

    # ── Step 10: Demo prediction on last processed symbol ─────────────────────
    log.info(f"\nStep 10: Demo prediction for {last_symbol}…")
    last_scaler = all_scalers.get(last_symbol, next(iter(all_scalers.values())))
    pred_return, confidence = predict_next_30_days(
        symbol        = last_symbol,
        model         = model,
        scaler_params = last_scaler,
        feature_cols  = feature_cols,
        market_ctx    = market_ctx,
        start         = args.start,
        end           = args.end,
        use_sentiment = not args.no_sentiment,
    )

    # Key drivers
    drivers = []
    if last_financial.get("piotroski_score_norm", 0) > 0.6:
        drivers.append(
            f"High Piotroski F-Score ({last_financial['piotroski_score_norm']*9:.0f}/9)"
        )
    if last_financial.get("returnOnEquity", 0) > 0.12:
        drivers.append(f"Strong ROE ({last_financial['returnOnEquity']:.1%})")
    if last_financial.get("altman_z_norm", 0) > 0.6:
        drivers.append("Healthy Altman Z-Score (safe zone)")
    if not np.isnan(pred_return) and pred_return > 0.03:
        drivers.append("Positive transformer momentum signal")
    if not drivers:
        drivers.append("Moderate composite signal")

    # ── Final output ──────────────────────────────────────────────────────────
    print()
    print("=" * 50)
    print("  TRIPLE-BRANCH HYBRID MODEL — TRAINING COMPLETE")
    print("=" * 50)
    print(f"  Symbols trained  : {', '.join(args.symbols)}")
    print(f"  Total sequences  : {len(X):,}")
    print()
    print(f"  Demo prediction for: {last_symbol}")
    if not np.isnan(pred_return):
        print(f"  Predicted 30-Day Return : {pred_return:+.1%}")
        print(f"  Confidence Score        : {confidence:.2f}")
    else:
        print("  Prediction              : N/A (insufficient data)")
    print()
    print("  Key Drivers:")
    for d in drivers:
        print(f"    - {d}")
    print()
    print(f"  Test Set Performance:")
    print(f"    MAE       : {mae:.5f}")
    print(f"    Hit Ratio : {hit:.1%}")
    print(f"    IC        : {float(ic):.4f}")
    print()
    print(f"  Model Saved As:")
    print(f"    {primary.name}  (for predict.py / ensemble / composite)")
    print(f"    {archive.name}  (versioned archive)")
    print("=" * 50)
    print()
    print("  Compatible with:")
    print(f"    python predict.py --model-name {args.model_name}")
    print(f"    python ensemble_predict.py --model-names {args.model_name}")
    print(f"    python composite_predict.py --model-name {args.model_name}")
    print()


if __name__ == "__main__":
    main()
