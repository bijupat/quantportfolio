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

# Piotroski/Altman fundamentals (market_data.services.quality) are fetched
# from yfinance's *current* financial statements — piotroski.py/altman.py
# both pull ticker.financials/balance_sheet/cashflow, which yfinance always
# returns as the company's latest reported fiscal years, regardless of what
# date is being computed for. There is no point-in-time historical snapshot
# available from this data source.
#
# Without bounding this, build_dataset_from_db's is_hybrid branch would
# broadcast a single "as of build time" score across the entire feat_df
# index — which can span a decade — feeding 2025 fundamentals into 2015
# training rows as if they were true then (bug #2: direct lookahead
# leakage). This constant bounds how far back that broadcast is allowed to
# reach before falling back to a neutral value instead.
#
# This is a mitigation, not a full fix: it shrinks the leakage window, it
# doesn't eliminate leakage within that window (the value used is still
# "whatever's true today," which may differ from what was knowable on any
# specific day within the window too, e.g. if a new filing landed partway
# through it). Fully eliminating this would require a point-in-time
# fundamentals data source, which is out of scope here. Tune down for a
# stricter (more neutral-filled, less leaky) dataset, or up for more "real"
# signal at the cost of a wider leakage window.
HYBRID_FUNDAMENTALS_RECENCY_WINDOW_DAYS = 90

# Neutral fill value for hybrid quality features outside the recency window
# above, and matches the existing "no data available" convention used
# elsewhere in this codebase (see quality.compute_composite_quality's
# p_norm fallback) — a genuinely unknown/inapplicable score, not an
# assertion that the stock is average.
NEUTRAL_QUALITY_FILL = 0.5


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
    scalers: Optional[Dict[str, Dict[str, Dict[str, float]]]] = None,
) -> Tuple[np.ndarray, np.ndarray, Dict, List[str]]:
    """
    Builds (X, y) sequence matrices using Django DB services as the data source.

    Args:
        symbols: Symbols to build sequences for.
        start_date: Start of the price/feature history window.
        end_date: End of the price/feature history window (inclusive).
        seq_len: Length of each input sequence window.
        horizon: Forward-return horizon in trading days. Only meaningful in
            mode="train" (used to compute labels); ignored for label purposes
            in mode="inference" since no forward label exists yet.
        use_sentiment: Whether to join sentiment features onto the feature matrix.
        is_hybrid: Whether to include hybrid-model fundamental feature columns.
            See HYBRID_FUNDAMENTALS_RECENCY_WINDOW_DAYS above for how
            lookahead exposure from quality scores is bounded, and the
            module docstring above for the [ts | financial | sentiment]
            column-ordering contract this branch must produce (see
            model_transformer.TripleBranchHybridModel.call()).
        mode: "train" (default) computes forward-return labels via
            create_labels() and drops the trailing `horizon` rows near
            end_date that have no future close to label against, and always
            derives fresh per-symbol/per-column normalization statistics
            from the full training window being built here — that's the
            correct source for scalers, since it's the distribution the
            model is about to learn against. "inference" skips label
            computation and keeps every row through end_date, returning a
            single window anchored at the actual most recent available
            date; see `scalers` below for how normalization is handled in
            this mode.
        scalers: Pre-computed per-symbol, per-column {"mean": ..., "std": ...}
            statistics, as produced by a prior mode="train" call and
            persisted on TrainedModel.scalers_json (see
            forecasting.services.predictor.get_or_predict_bulk, which passes
            trained_model.scalers_json through here). Only consulted when
            mode="inference" — mode="train" always computes its own fresh
            scalers regardless of this argument, since a training call *is*
            the process that produces this artifact in the first place.

            This is the fix for a real inference-correctness bug: a
            transformer's weights are calibrated against the exact
            mean/std each input feature had at training time. Previously,
            mode="inference" recomputed mean/std from scratch every call —
            but an inference call only ever has a short window in scope
            (seq_len + ~100 days for one symbol), not the multi-year
            training distribution, so the resulting normalization was
            arbitrary per symbol/run and had no relationship to what the
            model actually learned. This silently produced numerically
            wrong predictions with no error anywhere in the pipeline: shapes
            still matched, so nothing failed, it just fed the network inputs
            on the wrong scale. Confirmed via a side-by-side comparison
            against a legacy inference path that correctly reused its saved
            scalers.json — the two pipelines' raw transformer scores for the
            same symbols/model/date had ~0.12 correlation and a consistent
            negative bias, which is the expected signature of exactly this
            bug (short, idiosyncratic recent windows distort each symbol's
            mean/std differently and unpredictably relative to what the
            model was calibrated on).

            When `scalers` is supplied and mode="inference", each
            feature column is normalized using scalers[symbol.ticker][col]
            if present. If a symbol or column is missing from `scalers`
            (e.g. a newly-listed stock added to a universe after training,
            or a feature column added since the model was trained), that
            column falls back to being computed fresh from the available
            inference-window data, WITH a warning logged — this is a
            genuine degraded case worth knowing about, not silently
            swallowed, but it shouldn't hard-fail inference for the whole
            symbol over one missing column. When `scalers` is None in
            inference mode (e.g. a TrainedModel row saved before
            scalers_json was populated), behavior falls back to the
            previous recompute-from-window approach for every column, with
            a single warning logged once per call rather than per column.

    Returns:
        Tuple of (X, y, scalers_info, feature_columns). In mode="inference",
        X contains exactly one window per symbol and y is an unused
        zero-filled placeholder. scalers_info is always the scalers actually
        applied (whether freshly computed or reused from the `scalers` arg),
        so callers can inspect/persist what was really used.
    """
    if mode not in ("train", "inference"):
        raise ValueError(f"mode must be 'train' or 'inference', got {mode!r}")

    logger.info(f"=== DB-FIRST DATASET BUILD STARTED (mode={mode}) ===")

    if mode == "inference" and scalers is None:
        logger.warning(
            "build_dataset_from_db(mode='inference') called without saved `scalers` — "
            "falling back to computing normalization statistics from the short inference "
            "window itself. This reproduces the pre-fix inference-scaling bug for this "
            "call: predictions will not be normalized the same way the model was trained, "
            "and will likely be numerically wrong. Pass trained_model.scalers_json as "
            "`scalers` unless this TrainedModel genuinely predates scalers_json being "
            "populated (in which case, retrain to get a usable scalers artifact)."
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
            # ── Bug #2 mitigation: bounded lookahead for quality features ──
            # get_quality_scores_bulk returns ONE score per symbol, computed
            # from yfinance's current financials — see the
            # HYBRID_FUNDAMENTALS_RECENCY_WINDOW_DAYS docstring above for why
            # true point-in-time correctness isn't achievable here. Only
            # rows within the recency window of end_date get that real
            # value; older rows get a neutral fill instead of a
            # misattributed one, bounding (not eliminating) the leakage.
            qual_qs = get_quality_scores_bulk([symbol], end_date)
            qual_obj = qual_qs.first()
            p_norm = (qual_obj.piotroski_score / 9.0) if (qual_obj and qual_obj.piotroski_score) else 0.5
            a_norm = min(max(qual_obj.altman_z, 0.0), 5.0) / 5.0 if (qual_obj and qual_obj.altman_z) else 0.0

            recency_cutoff = pd.Timestamp(end_date) - pd.Timedelta(days=HYBRID_FUNDAMENTALS_RECENCY_WINDOW_DAYS)
            within_recency_window = feat_df.index >= recency_cutoff

            feat_df["returnOnEquity"] = 0.12
            feat_df["returnOnAssets"] = 0.05
            feat_df["trailingPE"] = 0.25
            feat_df["revenueGrowth"] = 0.10
            feat_df["debtToEquity"] = 0.15
            feat_df["freeCashflow_norm"] = 0.05
            feat_df["piotroski_score_norm"] = np.where(within_recency_window, p_norm, NEUTRAL_QUALITY_FILL)
            feat_df["altman_z_norm"] = np.where(within_recency_window, a_norm, NEUTRAL_QUALITY_FILL)

            # ── Bug #1 fix: explicit [ts | financial | sentiment] ordering ──
            # TripleBranchHybridModel.call() (model_transformer.py) slices
            # positionally: x[:, :, :n_ts], x[:, 0, n_ts:n_ts+n_fin],
            # x[:, :, n_ts+n_fin:]. That model code was always correct — the
            # bug was here: sentiment columns were joined onto feat_df
            # BEFORE this block runs, so the previous column order was
            # [ts | sentiment | financial], not [ts | financial | sentiment].
            # Every hybrid model trained before this fix learned on branches
            # fed scrambled data (part financial + part sentiment in each
            # slot) and should be considered invalid / retrained.
            ts_cols = [c for c in feat_df.columns if c not in FINANCIAL_FEATURES and c not in SENTIMENT_FEATURES]
            feat_df = feat_df[ts_cols + FINANCIAL_FEATURES + SENTIMENT_FEATURES]
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
        # mode="train": always derive fresh mean/std from this symbol's full
        # training-window feat_arr — this IS the correct source for scalers,
        # since it's the distribution the model is about to be fit against.
        #
        # mode="inference": reuse the saved training-time scalers whenever
        # available, rather than recomputing from the short inference
        # window (see the `scalers` parameter docstring above for why that
        # recompute was a correctness bug, not a stylistic difference).
        symbol_saved_scalers = (scalers or {}).get(symbol.ticker) if mode == "inference" else None
        missing_cols_in_saved_scalers: List[str] = []

        scaler_dict = {}
        feat_norm = feat_arr.copy().astype(np.float64)
        for i, col in enumerate(master_feature_cols):
            reused = symbol_saved_scalers.get(col) if symbol_saved_scalers else None
            if reused is not None:
                mu = float(reused["mean"])
                sigma = float(reused["std"])
            else:
                if mode == "inference" and scalers is not None:
                    # `scalers` was supplied but doesn't cover this symbol/column —
                    # a genuine partial-coverage case (new listing, or a feature
                    # added since training), not the "no scalers passed at all"
                    # case already warned about above. Track it for a single
                    # summarized warning after the column loop rather than
                    # logging once per column.
                    missing_cols_in_saved_scalers.append(col)
                col_series = clip_outliers(pd.Series(feat_arr[:, i]))
                mu = float(col_series.mean())
                sigma = float(col_series.std())
            sigma = sigma + 1e-9
            feat_norm[:, i] = (feat_arr[:, i] - mu) / sigma
            scaler_dict[col] = {"mean": mu, "std": sigma}

        if missing_cols_in_saved_scalers:
            logger.warning(
                f"[{symbol.ticker}] {len(missing_cols_in_saved_scalers)} feature column(s) "
                f"not found in saved scalers — fell back to computing them fresh from the "
                f"inference window: {missing_cols_in_saved_scalers}. This symbol's prediction "
                f"may be less reliable than one fully covered by the trained model's scalers."
            )

        scalers_info[symbol.ticker] = scaler_dict

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

    if mode == "train":
        shuffle_idx = np.random.permutation(len(X_all))
        X_all = X_all[shuffle_idx]
        y_all = y_all[shuffle_idx]

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
) -> TrainedModel:
    """
    Trains a QuantTransformer model on DB data and registers it in TrainedModel.
    """
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

    splits = walk_forward_splits(len(X), train_ratio=0.70, val_ratio=0.15)
    tr_sl, val_sl, te_sl = splits[0]
    X_train, y_train = X[tr_sl], y[tr_sl]
    X_val, y_val = X[val_sl], y[val_sl]
    X_test, y_test = X[te_sl], y[te_sl]

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

    NOTE: horizon is currently hardcoded to 30 below (not yet read from a
    --horizon CLI option, since train_hybrid_model.py has none) — deferred,
    same as before this pass; only bugs #1 (branch ordering) and #2
    (fundamentals lookahead) were in scope here.
    """
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

    splits = walk_forward_splits(len(X), train_ratio=0.70, val_ratio=0.15)
    tr_sl, val_sl, te_sl = splits[0]
    X_train, y_train = X[tr_sl], y[tr_sl]
    X_val, y_val = X[val_sl], y[val_sl]
    X_test, y_test = X[te_sl], y[te_sl]

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