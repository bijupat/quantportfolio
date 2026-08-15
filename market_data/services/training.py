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
    """30-day forward return label: (close[t+horizon] - close[t]) / close[t]"""
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
) -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
    """
    Builds (X, y) sequence matrices using Django DB services as the data source.
    """
    logger.info("=== DB-FIRST DATASET BUILD STARTED ===")
    
    # 1. Fetch benchmark market context (NIFTY + SENSEX, prefixed) from DB
    ctx_df = get_market_context_features(start_date, end_date)
    
    all_X, all_y = [], []
    scalers_info = {}
    master_feature_cols = []

    # 2. Iterate through requested symbols
    for symbol in symbols:
        logger.info(f"Processing DB training sequences for {symbol.ticker}...")

        # Price history from DB
        bars_qs = get_price_bars(symbol, start_date, end_date)
        raw_df = dataframe_from_bars(bars_qs)
        if raw_df.empty or len(raw_df) < (seq_len + horizon + 10):
            logger.warning(f"Insufficient price history in DB for {symbol.ticker}. Skipping.")
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

        # Hybrid model includes fundamental scores in feature matrix
        if is_hybrid:
            qual_qs = get_quality_scores_bulk([symbol], end_date)
            qual_obj = qual_qs.first()
            p_norm = (qual_obj.piotroski_score / 9.0) if (qual_obj and qual_obj.piotroski_score) else 0.5
            a_norm = min(max(qual_obj.altman_z, 0.0), 5.0) / 5.0 if (qual_obj and qual_obj.altman_z) else 0.0

            feat_df["returnOnEquity"] = 0.12
            feat_df["returnOnAssets"] = 0.05
            feat_df["trailingPE"] = 0.25
            feat_df["revenueGrowth"] = 0.10
            feat_df["debtToEquity"] = 0.15
            feat_df["freeCashflow_norm"] = 0.05
            feat_df["piotroski_score_norm"] = p_norm
            feat_df["altman_z_norm"] = a_norm

        # Labels (30-day forward return)
        labels = create_labels(raw_df["Close"], horizon=horizon)
        feat_df["__label__"] = labels.reindex(feat_df.index)

        # Clean NaNs
        feat_df.dropna(subset=["__label__"], inplace=True)
        feat_df.ffill().bfill().fillna(0.0, inplace=True)

        if len(feat_df) < (seq_len + 10):
            continue

        # Column ordering (preserve __label__ during alignment)
        current_cols = [c for c in feat_df.columns if c != "__label__"]
        if not master_feature_cols:
            master_feature_cols = current_cols
        else:
            for col in master_feature_cols:
                if col not in feat_df.columns:
                    feat_df[col] = 0.0
            
            # Keep the target label safe while forcing column alignment
            label_backup = feat_df["__label__"].copy()
            feat_df = feat_df[master_feature_cols]
            feat_df["__label__"] = label_backup

        feat_arr = feat_df[master_feature_cols].values.astype(np.float32)
        label_arr = feat_df["__label__"].values.astype(np.float32)

        # Z-score normalization per symbol
        scaler_dict = {}
        feat_norm = feat_arr.copy().astype(np.float64)
        for i, col in enumerate(master_feature_cols):
            col_series = clip_outliers(pd.Series(feat_arr[:, i]))
            mu = float(col_series.mean())
            sigma = float(col_series.std()) + 1e-9
            feat_norm[:, i] = (feat_arr[:, i] - mu) / sigma
            scaler_dict[col] = {"mean": mu, "std": sigma}

        scalers_info[symbol.ticker] = scaler_dict

        # Build sliding windows
        X_sym, y_sym = build_sequences(feat_norm.astype(np.float32), label_arr, seq_len=seq_len)
        if len(X_sym) > 0:
            all_X.append(X_sym)
            all_y.append(y_sym)

    if not all_X:
        raise RuntimeError("Failed to build sequences from DB. Ensure price history is fetched.")

    X_all = np.concatenate(all_X, axis=0)
    y_all = np.concatenate(all_y, axis=0)

    # Cross-stock shuffle
    shuffle_idx = np.random.permutation(len(X_all))
    X_all = X_all[shuffle_idx]
    y_all = y_all[shuffle_idx]

    logger.info(f"DB Dataset created: X={X_all.shape}, y={y_all.shape}")
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
) -> TrainedModel:
    """
    Trains a QuantTransformer model on DB data and registers it in TrainedModel.
    """
    # 1. Dataset
    X, y, scalers_info, feature_cols = build_dataset_from_db(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        seq_len=seq_len,
        horizon=horizon,
        use_sentiment=use_sentiment,
        is_hybrid=False,
    )

    n_features = X.shape[2]

    # 2. Walk-forward split
    splits = walk_forward_splits(len(X), train_ratio=0.70, val_ratio=0.15)
    tr_sl, val_sl, te_sl = splits[0]
    X_train, y_train = X[tr_sl], y[tr_sl]
    X_val, y_val = X[val_sl], y[val_sl]
    X_test, y_test = X[te_sl], y[te_sl]

    # 3. Keras Build & Train
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

    # 4. Evaluation
    eval_metrics = keras_evaluate_model(model, X_test, y_test)

    # 5. Architecture metadata
    arch_data = {
        "seq_len": seq_len,
        "n_features": n_features,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "model_type": TrainedModel.STANDARD,
    }

    # 6. Save weights to temporary file & register in DB
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

        trained_model.arch_json = arch_data
        trained_model.scalers_json = scalers_info
        trained_model.eval_json = eval_metrics
        trained_model.save()

    finally:
        if os.path.exists(tmp_weights_path):
            os.remove(tmp_weights_path)

    logger.info(f"Successfully trained and registered TrainedModel: {model_name}")
    return trained_model


def train_hybrid_model_service(
    model_name: str,
    symbols: List[Symbol],
    start_date: date,
    end_date: date,
    universe: Optional[Universe] = None,
    seq_len: int = 120,
    epochs: int = 60,
    batch_size: int = 32,
    d_model: int = 64,
    n_heads: int = 4,
    n_layers: int = 2,
    lstm_units: int = 64,
    use_sentiment: bool = True,
) -> TrainedModel:
    """
    Trains a TripleBranchHybridModel on DB data and registers it in TrainedModel.
    """
    # 1. Dataset
    X, y, scalers_info, feature_cols = build_dataset_from_db(
        symbols=symbols,
        start_date=start_date,
        end_date=end_date,
        seq_len=seq_len,
        horizon=30,
        use_sentiment=use_sentiment,
        is_hybrid=True,
    )

    n_ts_features = len([c for c in feature_cols if c not in FINANCIAL_FEATURES + SENTIMENT_FEATURES])
    n_fin_features = len([c for c in feature_cols if c in FINANCIAL_FEATURES])
    n_sent_features = len([c for c in feature_cols if c in SENTIMENT_FEATURES])
    n_total = X.shape[2]

    # 2. Split
    splits = walk_forward_splits(len(X), train_ratio=0.70, val_ratio=0.15)
    tr_sl, val_sl, te_sl = splits[0]
    X_train, y_train = X[tr_sl], y[tr_sl]
    X_val, y_val = X[val_sl], y[val_sl]
    X_test, y_test = X[te_sl], y[te_sl]

    # 3. Keras Build & Train
    model = build_hybrid_model(
        seq_len=seq_len,
        n_ts_features=n_ts_features,
        n_financial_features=n_fin_features,
        n_sentiment_features=n_sent_features,
        d_model=d_model,
        n_heads=n_heads,
        n_layers=n_layers,
        lstm_units=lstm_units,
    )

    keras_train_hybrid(
        model=model, X_train=X_train, y_train=y_train,
        X_val=X_val, y_val=y_val, model_name=model_name,
        epochs=epochs, batch_size=batch_size,
    )

    # 4. Evaluation
    y_pred = model.predict(X_test, verbose=0).flatten()
    from scipy.stats import spearmanr
    mse = float(np.mean((y_test - y_pred) ** 2))
    mae = float(np.mean(np.abs(y_test - y_pred)))
    hit = float(np.mean(np.sign(y_test) == np.sign(y_pred)))
    ic, _ = spearmanr(y_test, y_pred)

    eval_metrics = {
        "MSE": mse, "MAE": mae, "HitRatio": hit,
        "IC_Spearman": float(ic) if np.isfinite(ic) else 0.0,
    }

    # 5. Architecture metadata
    arch_data = {
        "seq_len": seq_len,
        "n_features": n_total,
        "d_model": d_model,
        "n_heads": n_heads,
        "n_layers": n_layers,
        "n_ts_features": n_ts_features,
        "n_financial_features": n_fin_features,
        "n_sentiment_features": n_sent_features,
        "model_type": TrainedModel.HYBRID,
    }

    # 6. Save weights to temporary file & register in DB
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
                "model_type": TrainedModel.HYBRID,
                "seq_len": seq_len,
                "d_model": d_model,
                "n_heads": n_heads,
                "n_layers": n_layers,
                "n_features": n_total,
                "arch_json": arch_data,
                "scalers_json": scalers_info,
                "eval_json": eval_metrics,
                "trained_on": universe,
            }
        )

        with open(tmp_weights_path, "rb") as f_in:
            trained_model.weights_file.save(f"{model_name}.weights.h5", File(f_in), save=True)

        trained_model.arch_json = arch_data
        trained_model.scalers_json = scalers_info
        trained_model.eval_json = eval_metrics
        trained_model.save()

    finally:
        if os.path.exists(tmp_weights_path):
            os.remove(tmp_weights_path)

    logger.info(f"Successfully trained and registered Hybrid TrainedModel: {model_name}")
    return trained_model