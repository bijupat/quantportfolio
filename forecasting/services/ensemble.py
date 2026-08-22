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
) -> Tuple[Dict[str, float], Dict[str, Dict[str, float]], Dict[str, List[str]]]:
    """
    Runs multiple AI models and averages their predictions.

    Each model may fail to produce a prediction for a given symbol (e.g.
    insufficient price history for that model's seq_len). Previously, a
    symbol's weighted average silently renormalised over only the models
    that *did* score it — so a symbol scored by 1 of 3 models got 100% of
    its final score from that one model, indistinguishable in the output
    from a symbol that all 3 models agreed on. This function now tracks and
    surfaces that per-symbol model coverage explicitly, rather than only
    computing a number and hoping nobody needed to know how it was built.

    Returns:
        A 3-tuple of:
          - weighted_scores_map: {ticker: final weighted-average score},
            renormalised over only the models that scored that symbol.
          - per_model_raw_scores_map: {model_name: {ticker: raw_score}}.
          - contributing_models_map: {ticker: [model_name, ...]} — which
            models actually contributed a score for that symbol. Any ticker
            whose list is shorter than `models_with_weights` was scored by
            a strict subset of the requested ensemble; callers (e.g.
            forecasting.services.composite.compute_composite_scores) should
            treat these as lower-confidence than a symbol scored by every
            model, since the "weighted average" is not comparable across
            symbols with different coverage.
    """
    total_w = sum(models_with_weights.values())
    if total_w == 0:
        raise ValueError("Total ensemble weights cannot be zero.")

    norm_w = {m: w / total_w for m, w in models_with_weights.items()}
    full_model_names = [m.name for m in norm_w]
    n_models_requested = len(norm_w)

    weighted_sums: Dict[str, float] = {}
    weight_totals: Dict[str, float] = {}
    per_model_scores: Dict[str, Dict[str, float]] = {}
    contributing_models: Dict[str, List[str]] = {}

    for trained_model, w in norm_w.items():
        logger.info(f"Ensemble inference: '{trained_model.name}' weight={w:.2f}")
        per_model_scores[trained_model.name] = {}

        # Utilize the DB-first predictor service
        predictions_map = get_or_predict_bulk(symbols, trained_model, as_of, use_sentiment)

        scored_ids = {sym_id for sym_id in predictions_map}
        requested_ids = {s.id for s in symbols}
        missing_ids = requested_ids - scored_ids
        if missing_ids:
            missing_tickers = [s.ticker for s in symbols if s.id in missing_ids]
            logger.warning(
                f"Model '{trained_model.name}' failed to score {len(missing_tickers)} of "
                f"{len(symbols)} requested symbols (likely insufficient history): "
                f"{missing_tickers}"
            )

        for sym_id, pred in predictions_map.items():
            sym_ticker = pred.symbol.ticker
            score = pred.predicted_return

            per_model_scores[trained_model.name][sym_ticker] = score
            weighted_sums[sym_ticker] = weighted_sums.get(sym_ticker, 0.0) + w * score
            weight_totals[sym_ticker] = weight_totals.get(sym_ticker, 0.0) + w
            contributing_models.setdefault(sym_ticker, []).append(trained_model.name)

    final_scores = {
        sym: weighted_sums[sym] / weight_totals[sym]
        for sym in weighted_sums if weight_totals[sym] > 0
    }

    # Surface any symbol whose final score was built from a strict subset of
    # the requested ensemble — this is the actual fix: previously nothing
    # downstream could tell a fully-agreed score from a single-model
    # fallback, since both looked like an ordinary float in final_scores.
    partial_coverage = {
        sym: models for sym, models in contributing_models.items()
        if len(models) < n_models_requested
    }
    if partial_coverage:
        logger.warning(
            f"{len(partial_coverage)} of {len(final_scores)} symbols were scored by fewer "
            f"than all {n_models_requested} requested models — their weighted average is "
            f"not on the same footing as fully-covered symbols. "
            f"Requested models: {full_model_names}. Partial coverage: {partial_coverage}"
        )

    return final_scores, per_model_scores, contributing_models