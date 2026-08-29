import os
import json
import logging
import tempfile
import numpy as np
import pandas as pd
from datetime import date, datetime
from pathlib import Path
from typing import List, Tuple, Dict, Optional

import tensorflow as tf
from django.db import transaction
from django.core.files import File

from core.models import Symbol, Universe
from market_data.models import TrainedModel, TrainingRun
from market_data.services.prices import get_price_bars, dataframe_from_bars
from market_data.services.indicators import get_indicators
from market_data.services.sentiment import get_sentiment
from market_data.services.quality import get_quality_scores_bulk
from market_data.services.market_context import get_market_context_features
from model_transformer import (
    build_transformer_model,
    train_model as keras_train_model,
    evaluate_model as keras_evaluate_model,
    build_hybrid_model,
    train_hybrid_model as keras_train_hybrid,
)

from portfolio_optimizer import walk_forward_splits
from market_data.services.indicators import (
    FEATURE_COLUMNS, MARKET_CONTEXT_COLS, FINANCIAL_FEATURES, SENTIMENT_FEATURES,
)
from util import clip_outliers, build_sequences

logger = logging.getLogger(__name__)

def create_labels(close: pd.Series, horizon: int = 30) -> pd.Series:
    """Forward return label over `horizon` trading days: (close[t+horizon] - close[t]) / close[t]"""
    future_close = close.shift(-horizon)
    return (future_close - close) / (close + 1e-9)

# ---------------------------------------------------------------------------
# DB-First Dataset Builder
# ---------------------------------------------------------------------------

def build_dataset_from_db(
    symbols: List[Symbol],
    start_date: date,
    end_date: date,
    seq_len: int = 60,
    horizon: int = 30,
    use_sentiment: bool = True,
    is_hybrid: bool = False,
    mode: str = "train",
    scalers: Optional[Dict[str, Dict[str, float]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
    """
    Builds (X, y) sequence matrices using Django DB services as the data source.

    Args:
        symbols: Symbols to build sequences for.
        start_date: Start of the price/feature history window.
        end_date: End of the price/feature history window (inclusive).
        seq_len: Length of each input sequence window.
        horizon: Forward-return horizon in trading days.
        use_sentiment: Whether to join sentiment features onto the feature matrix.
        is_hybrid: Determines feature ordering. Fundamentals support removed due to lookahead bias.
        mode: "train" (default) or "inference". In "train" mode, data is returned UNSCALED.
            Scaling must happen after walk_forward_splits. In "inference" mode, the provided
            `scalers` dict is applied.
        scalers: Pre-computed global per-column {"mean": ..., "std": ...} statistics.
            Only consulted when mode="inference".
    """
    if mode not in ("train", "inference"):
        raise ValueError(f"mode must be 'train' or 'inference', got {mode!r}")

    logger.info(f"=== DB-FIRST DATASET BUILD STARTED (mode={mode}) ===")

    if mode == "inference" and scalers is None:
        logger.warning(
            "build_dataset_from_db(mode='inference') called without saved `scalers` — "
            "falling back to computing normalization statistics from the short inference "
            "window itself. This will distort predictions."
        )

    # 1. Fetch benchmark market context (NIFTY + SENSEX, prefixed) from DB
    ctx_df = get_market_context_features(start_date, end_date)
    
    all_X, all_y = [], []
    scalers_info = {}
    master_feature_cols = []

    # 2. Iterate through requested symbols
    for symbol in symbols:
        logger.info(f"Processing DB {mode} sequences for {symbol.ticker}...")

        # Price history from DB
        bars_qs = get_price_bars(symbol, start_date, end_date)
        raw_df = dataframe_from_bars(bars_qs)

        min_bars_required = (seq_len + 10) if mode == "inference" else (seq_len + horizon + 10)
        if raw_df.empty or len(raw_df) < min_bars_required:
            logger.warning(f"Insufficient price history in DB for {symbol.ticker} (mode={mode}). Skipping.")
            continue

        # Technical indicators from DB
        ind_qs = get_indicators(symbol, bars_qs)
        ind_data = list(ind_qs.values("date", "values"))
        if not ind_data:
            continue
        feat_df = pd.DataFrame([{"Date": pd.to_datetime(d["date"]), **d["values"]} for d in ind_data])
        feat_df.set_index("Date", inplace=True)

        # Merge Market Context
        if not ctx_df.empty:
            feat_df = feat_df.join(ctx_df, how="left").ffill().bfill()

        # Merge Sentiment
        if use_sentiment:
            sent_qs = get_sentiment(symbol, start_date, end_date)
            sent_data = list(sent_qs.values("date", "sentiment_score", "positive_count", "negative_count", "news_volume"))
            if sent_data:
                sent_df = pd.DataFrame(sent_data)
                sent_df.rename(columns={"date": "Date"}, inplace=True)
                sent_df["Date"] = pd.to_datetime(sent_df["Date"])
                sent_df.set_index("Date", inplace=True)
                feat_df = feat_df.join(sent_df, how="left")
            
        for c in ["sentiment_score", "positive_count", "negative_count", "news_volume"]:
            if c not in feat_df.columns:
                feat_df[c] = 0.0
        feat_df.fillna(0.0, inplace=True)

        if is_hybrid:
            raise NotImplementedError(
                "Hybrid fundamentals lookup disabled due to severe lookahead bias. "
                "Point-in-time fundamentals database required before re-enabling."
            )
        else:
            # Explicit feature-column enforcement (standard/non-hybrid path).
            expected_cols = FEATURE_COLUMNS + MARKET_CONTEXT_COLS
            for col in expected_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            feat_df = feat_df[expected_cols]

        if mode == "train":
            labels = create_labels(raw_df["Close"], horizon=horizon)
            feat_df["__label__"] = labels.reindex(feat_df.index)
            feat_df.dropna(subset=["__label__"], inplace=True)
        else:
            feat_df["__label__"] = 0.0

        feat_df = feat_df.ffill().bfill()
        feat_df.fillna(0.0, inplace=True)

        min_len_required = seq_len if mode == "inference" else (seq_len + 10)
        if len(feat_df) < min_len_required:
            continue

        # Column ordering (preserve __label__ during alignment)
        current_cols = [c for c in feat_df.columns if c != "__label__"]
        if not master_feature_cols:
            master_feature_cols = current_cols
        else:
            for col in master_feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            
            label_backup = feat_df["__label__"].copy()
            feat_df = feat_df[master_feature_cols]
            feat_df["__label__"] = label_backup

        feat_arr = feat_df[master_feature_cols].values.astype(np.float32)
        label_arr = feat_df["__label__"].values.astype(np.float32)

        # ── Normalization ────────────────────────────────────────────────
        if mode == "train":
            # Lookahead Leakage Fix: Do NOT scale data here before splitting.
            # Pass raw unscaled features to build_sequences.
            feat_norm = feat_arr.copy().astype(np.float32)
        else:
            # Inference mode: apply the globally fit scalers from X_train
            missing_cols_in_saved_scalers: List[str] = []
            feat_norm = feat_arr.copy().astype(np.float64)
            for i, col in enumerate(master_feature_cols):
                reused = (scalers or {}).get(col)
                if reused is not None:
                    mu = float(reused["mean"])
                    sigma = float(reused["std"])
                else:
                    if scalers is not None:
                        missing_cols_in_saved_scalers.append(col)
                    # Fallback to computing on inference window
                    col_series = clip_outliers(pd.Series(feat_arr[:, i]))
                    mu = float(col_series.mean())
                    sigma = float(col_series.std())
                sigma = sigma + 1e-9
                feat_norm[:, i] = (feat_arr[:, i] - mu) / sigma
                
            if missing_cols_in_saved_scalers:
                logger.warning(
                    f"[{symbol.ticker}] {len(missing_cols_in_saved_scalers)} feature column(s) "
                    f"not found in saved scalers. Fell back to inference window: "
                    f"{missing_cols_in_saved_scalers}."
                )

        if mode == "train":
            X_sym, y_sym = build_sequences(feat_norm.astype(np.float32), label_arr, seq_len=seq_len)
        else:
            X_last = feat_norm[-seq_len:].astype(np.float32)
            X_sym = X_last[np.newaxis, ...]
            y_sym = np.zeros((1,), dtype=np.float32)

        if len(X_sym) > 0:
            all_X.append(X_sym)
            all_y.append(y_sym)

    if not all_X:
        raise RuntimeError("Failed to build sequences from DB. Ensure price history is fetched.")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # REMOVED SHUFFLE HERE to prevent temporal leakage before walk-forward splits.

    logger.info(f"DB Dataset created (mode={mode}): X={X_all.shape}, y={y_all.shape}")
    return X_all, y_all, scalers_info, master_feature_cols


# ---------------------------------------------------------------------------
# Training Orchestrators (Standard & Hybrid)
# ---------------------------------------------------------------------------

def train_standard_model_service(
    model_name: str,
    symbols: List[Symbol],
    start_date: date,
    end_date: date,
    universe: Optional[Universe] = None,
    seq_len: int = 60,
    horizon: int = 30,
    epochs: int = 50,
    batch_size: int = 64,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    use_sentiment: bool = True,
    # top_n: int = 10,
    # run_backtest: bool = True,
) -> TrainedModel:
    """
    Trains a QuantTransformer model on DB data and registers it in TrainedModel.
    """
    X, y, _, feature_cols = build_dataset_from_db(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        seq_len=seq_len,
        horizon=horizon,
        use_sentiment=use_sentiment,
        is_hybrid=False,
        mode="train"
    )

    n_features = X.shape[2]

    splits = walk_forward_splits(len(X), train_ratio=0.70, val_ratio=0.15)
    tr_sl, val_sl, te_sl = splits[0]
    # Use .copy() to prevent modifying the original X array via slice views
    X_train, y_train = X[tr_sl].copy(), y[tr_sl].copy()
    X_val, y_val = X[val_sl].copy(), y[val_sl].copy()
    X_test, y_test = X[te_sl].copy(), y[te_sl].copy()

    # --- Lookahead Leakage Fix: Fit Global Scaler ONLY on X_train ---
    scalers_info = {}
    for i, col in enumerate(feature_cols):
        feature_slice = X_train[:, :, i].flatten()
        mu = float(np.mean(feature_slice))
        sigma = float(np.std(feature_slice)) + 1e-9
        
        scalers_info[col] = {"mean": mu, "std": sigma}
        
        X_train[:, :, i] = (X_train[:, :, i] - mu) / sigma
        X_val[:, :, i] = (X_val[:, :, i] - mu) / sigma
        X_test[:, :, i] = (X_test[:, :, i] - mu) / sigma

    # Shuffle training data AFTER chronological walk-forward splitting and scaling
    shuffle_idx = np.random.permutation(len(X_train))
    X_train = X_train[shuffle_idx]
    y_train = y_train[shuffle_idx]

    ff_dim = 4 * d_model  


    model = build_transformer_model(
        seq_len=seq_len,
        n_features=n_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        dropout=0.15,
        lr=1e-4,
    )

    keras_train_model(
        model, X_train, y_train, X_val, y_val,
        epochs=epochs, batch_size=batch_size, model_name=model_name,
    )

    eval_metrics = keras_evaluate_model(model, X_test, y_test)

   

    arch_data = {
        "seq_len": seq_len,
        "n_features": n_features,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "ff_dim": ff_dim, 
        "horizon": horizon,
        "model_type": TrainedModel.STANDARD,
    }

    with tempfile.NamedTemporaryFile(suffix=".weights.h5", delete=False) as tmp_file:
        tmp_weights_path = tmp_file.name

    try:
        import h5py
        with h5py.File(tmp_weights_path, "w") as f:
            for i, w in enumerate(model.weights):
                f.create_dataset(f"weight_{i:04d}", data=np.array(w))
            for k, v in arch_data.items():
                if isinstance(v, (int, float, str)):
                    f.attrs[k] = v

        trained_model, _ = TrainedModel.objects.get_or_create(
            name=model_name,
            defaults={
                "model_type": TrainedModel.STANDARD,
                "seq_len": seq_len,
                "horizon": horizon,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "n_features": n_features,
                "arch_json": arch_data,
                "scalers_json": scalers_info,
                "eval_json": eval_metrics,
                "trained_on": universe,
            }
        )

        with open(tmp_weights_path, "rb") as f_in:
            trained_model.weights_file.save(f"{model_name}.weights.h5", File(f_in), save=True)

        trained_model.horizon = horizon
        trained_model.arch_json = arch_data
        trained_model.scalers_json = scalers_info
        trained_model.eval_json = eval_metrics
        trained_model.save()

    finally:
        if os.path.exists(tmp_weights_path):
            os.remove(tmp_weights_path)

    logger.info(f"Successfully trained and registered TrainedModel: {model_name}")
    return trained_model


def train_hybrid_model_service(*args, **kwargs) -> TrainedModel:
    """
    Disabled due to fundamental data lookahead leakage.
    Requires point-in-time financial statements to safely train.
    """
    logger.error("train_hybrid_model_service is disabled to prevent lookahead leakage.")
    raise NotImplementedError("Hybrid fundamentals lookup requires a point-in-time financial database.")