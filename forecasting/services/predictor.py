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

def get_or_predict_bulk(symbols: list[Symbol], trained_model: TrainedModel, as_of: date, use_sentiment: bool = True) -> dict:
    """
    Bulk inference: Checks DB for cached predictions, runs Keras inference for missing ones.
    Returns a dict mapping symbol IDs to their Prediction objects.
    """
    symbol_ids = [s.id for s in symbols]
    
    # 1. Check Cache
    cached_qs = Prediction.objects.filter(symbol_id__in=symbol_ids, model=trained_model, as_of_date=as_of)
    cached_map = {p.symbol_id: p for p in cached_qs}
    
    missing_symbols = [s for s in symbols if s.id not in cached_map]
    
    if not missing_symbols:
        logger.info("All predictions found in cache.")
        return cached_map
        
    logger.info(f"Running AI inference for {len(missing_symbols)} symbols using '{trained_model.name}'...")
    
    # 2. Load the Keras Model
    keras_model = trained_model.build_keras_model()
    is_hybrid = trained_model.model_type == TrainedModel.HYBRID

    predictions_to_create = []

    # 3. Process missing symbols
    for symbol in missing_symbols:
        try:
            # Fetch a slightly longer window to accommodate seq_len and rolling indicators
            start_history = as_of - pd.Timedelta(days=trained_model.seq_len + 100)
            
            # Re-use our DB feature builder for inference (returns normalized features)
            X_arr, _, scalers_info, _ = build_dataset_from_db(
                symbols=[symbol],
                start_date=start_history,
                end_date=as_of,
                seq_len=trained_model.seq_len,
                horizon=30,
                use_sentiment=use_sentiment,
                is_hybrid=is_hybrid
            )

            if len(X_arr) == 0:
                logger.warning(f"Insufficient data to predict {symbol.ticker}")
                continue

            # Take the very last sequence window (the most recent one up to 'as_of')
            latest_window = X_arr[-1]
            x_tensor = tf.constant(latest_window[np.newaxis], dtype=tf.float32)
            
            # Forward pass
            pred_val = float(keras_model(x_tensor, training=False).numpy().flatten()[0])
            
            # Confidence heuristic (matches your hybrid_main.py logic)
            confidence = float(1 / (1 + np.exp(-abs(pred_val) * 20)))

            predictions_to_create.append(
                Prediction(
                    symbol=symbol,
                    model=trained_model,
                    as_of_date=as_of,
                    predicted_return=pred_val,
                    confidence=confidence
                )
            )

        except Exception as e:
            logger.error(f"Inference failed for {symbol.ticker}: {e}")

    # 4. Save to Database
    if predictions_to_create:
        with transaction.atomic():
            Prediction.objects.bulk_create(predictions_to_create, ignore_conflicts=True)
            
        # Update the cache map with newly created predictions
        new_qs = Prediction.objects.filter(
            symbol_id__in=[p.symbol_id for p in predictions_to_create], 
            model=trained_model, 
            as_of_date=as_of
        )
        for p in new_qs:
            cached_map[p.symbol_id] = p

    return cached_map