import logging
from datetime import date
from typing import Dict, List, Tuple
from core.models import Symbol
from market_data.models import TrainedModel
from forecasting.services.predictor import get_or_predict_bulk

logger = logging.getLogger(__name__)

def run_ensemble_layer(
    symbols: List[Symbol],
    models_with_weights: Dict[TrainedModel, float],
    as_of: date,
    use_sentiment: bool = True
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]]]:
    """
    Runs multiple AI models and averages their predictions.
    Returns: (weighted_scores_map, per_model_raw_scores_map)
    """
    total_w = sum(models_with_weights.values())
    if total_w == 0:
        raise ValueError("Total ensemble weights cannot be zero.")
        
    norm_w = {m: w / total_w for m, w in models_with_weights.items()}

    weighted_sums = {}
    weight_totals = {}
    per_model_scores = {}

    for trained_model, w in norm_w.items():
        logger.info(f"Ensemble inference: '{trained_model.name}' weight={w:.2f}")
        per_model_scores[trained_model.name] = {}
        
        # Utilize the DB-first predictor service
        predictions_map = get_or_predict_bulk(symbols, trained_model, as_of, use_sentiment)
        
        for sym_id, pred in predictions_map.items():
            sym_ticker = pred.symbol.ticker
            score = pred.predicted_return
            
            per_model_scores[trained_model.name][sym_ticker] = score
            weighted_sums[sym_ticker] = weighted_sums.get(sym_ticker, 0.0) + w * score
            weight_totals[sym_ticker] = weight_totals.get(sym_ticker, 0.0) + w

    final_scores = {
        sym: weighted_sums[sym] / weight_totals[sym]
        for sym in weighted_sums if weight_totals[sym] > 0
    }
    
    return final_scores, per_model_scores