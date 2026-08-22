import logging
import numpy as np
import pandas as pd
from datetime import date
from django.db import transaction
import tensorflow as tf

from core.models import Symbol
from market_data.models import TrainedModel
from forecasting.models import Prediction
from market_data.services.training import build_dataset_from_db

logger = logging.getLogger(__name__)

# Bump this whenever a change to the inference/feature-building pipeline
# would make previously cached Prediction rows numerically wrong or stale
# (e.g. the mode="inference" staleness fix in
# market_data.services.training.build_dataset_from_db, or any future change
# to feature columns, normalization, or sequence construction). Cosmetic or
# purely-additive changes (new logging, docstrings) don't need a bump.
#
# get_or_predict_bulk only treats a cached Prediction row as a genuine cache
# hit when its pipeline_version matches this constant — see the Prediction
# model's docstring in forecasting/models.py for the full rationale. This is
# the fix for the cache-staleness bug: previously Prediction rows were cached
# forever per (symbol, model, as_of_date) with no way to tell a row computed
# under old, buggy logic apart from one computed after a fix, so both were
# served from cache identically and indefinitely.
PIPELINE_VERSION = "2026.08-inference-mode-fix"


def get_or_predict_bulk(symbols: list[Symbol], trained_model: TrainedModel, as_of: date, use_sentiment: bool = True) -> dict:
    """
    Bulk inference: Checks DB for cached predictions, runs Keras inference for missing ones.

    A cached Prediction row only counts as a cache hit if its
    pipeline_version matches the current PIPELINE_VERSION. Rows from an
    older pipeline version (including pre-existing rows that default to
    "unknown") are treated as missing — they get recomputed and the stale
    row is replaced (via update_or_create) with a fresh one stamped with
    the current version, rather than being silently served or silently
    left un-overwritten by bulk_create's ignore_conflicts.

    Returns a dict mapping symbol IDs to their Prediction objects.
    """
    symbol_ids = [s.id for s in symbols]
    horizon = trained_model.horizon

    # 1. Check Cache — only rows matching the current pipeline version count.
    cached_qs = Prediction.objects.filter(
        symbol_id__in=symbol_ids,
        model=trained_model,
        as_of_date=as_of,
        pipeline_version=PIPELINE_VERSION,
    )
    cached_map = {p.symbol_id: p for p in cached_qs}

    # Surface how many rows exist for this (model, as_of_date) but were
    # rejected as stale, so a person watching logs can see the cache is
    # actively being refreshed rather than wondering why "cached" symbols
    # are being recomputed.
    stale_count = Prediction.objects.filter(
        symbol_id__in=symbol_ids,
        model=trained_model,
        as_of_date=as_of,
    ).exclude(pipeline_version=PIPELINE_VERSION).count()
    if stale_count:
        logger.info(
            f"Found {stale_count} cached Prediction row(s) for '{trained_model.name}' "
            f"on {as_of} from an older pipeline version — treating as stale and recomputing."
        )

    missing_symbols = [s for s in symbols if s.id not in cached_map]

    if not missing_symbols:
        logger.info("All predictions found in cache (current pipeline version).")
        return cached_map

    logger.info(f"Running AI inference for {len(missing_symbols)} symbols using '{trained_model.name}' (horizon={horizon}d)...")

    # 2. Load the Keras Model
    keras_model = trained_model.build_keras_model()
    is_hybrid = trained_model.model_type == TrainedModel.HYBRID

    predictions_to_write = []

    # 3. Process missing symbols
    for symbol in missing_symbols:
        try:
            # Fetch a slightly longer window to accommodate seq_len and rolling indicators
            start_history = as_of - pd.Timedelta(days=trained_model.seq_len + 100)

            # mode="inference" — returns the single window ending at the most
            # recent available date up to `as_of`, with no forward-label
            # requirement.
            X_arr, _, scalers_info, _ = build_dataset_from_db(
                symbols=[symbol],
                start_date=start_history,
                end_date=as_of,
                seq_len=trained_model.seq_len,
                horizon=horizon,
                use_sentiment=use_sentiment,
                is_hybrid=is_hybrid,
                mode="inference",
            )

            if len(X_arr) == 0:
                logger.warning(f"Insufficient data to predict {symbol.ticker}")
                continue

            # In mode="inference" this is the (only) window, correctly
            # anchored at the most recent available trading date <= as_of.
            latest_window = X_arr[-1]
            x_tensor = tf.constant(latest_window[np.newaxis], dtype=tf.float32)

            # Forward pass
            pred_val = float(keras_model(x_tensor, training=False).numpy().flatten()[0])

            # Confidence heuristic (matches your hybrid_main.py logic)
            confidence = float(1 / (1 + np.exp(-abs(pred_val) * 20)))

            predictions_to_write.append({
                "symbol": symbol,
                "predicted_return": pred_val,
                "confidence": confidence,
            })

        except Exception as e:
            logger.error(f"Inference failed for {symbol.ticker}: {e}")

    # 4. Save to Database — update_or_create per row (not bulk_create with
    # ignore_conflicts) so a stale row with the same (symbol, model,
    # as_of_date) is overwritten with the freshly computed value and the
    # current pipeline_version, rather than the unique_together constraint
    # silently blocking the write and leaving the stale row untouched.
    if predictions_to_write:
        with transaction.atomic():
            for item in predictions_to_write:
                fresh_pred, _created = Prediction.objects.update_or_create(
                    symbol=item["symbol"],
                    model=trained_model,
                    as_of_date=as_of,
                    defaults={
                        "horizon_days": horizon,
                        "predicted_return": item["predicted_return"],
                        "confidence": item["confidence"],
                        "pipeline_version": PIPELINE_VERSION,
                    },
                )
                cached_map[item["symbol"].id] = fresh_pred

    return cached_map