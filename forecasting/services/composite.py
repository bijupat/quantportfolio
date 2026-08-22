import hashlib
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Tuple

from django.db.models import QuerySet

from core.models import Symbol
from market_data.services.prices import get_price_bars, dataframe_from_bars
from market_data.services.quality import get_quality_scores_bulk
from forecasting.models import CompositeScore

logger = logging.getLogger(__name__)

# How far back to look for a usable closing price when building portfolio
# allocations. Generous enough to survive a short run of missing bars
# (e.g. a stale fetch_prices run) without silently returning price=0.0.
PRICE_LOOKUP_WINDOW_DAYS = 10


def _minmax_normalise(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a score dict to 0.0-1.0."""
    if not scores:
        return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo:
        return {sym: 0.5 for sym in scores}
    return {sym: float((v - lo) / (hi - lo)) for sym, v in scores.items()}


def get_technical_scores(symbols: List[Symbol], as_of: date) -> Tuple[Dict[str, float], Dict[str, bool]]:
    """Computes TSI (True Strength Index) from DB PriceBars and maps to 0.0-1.0.

    Returns:
        A 2-tuple of (scores, is_neutral_fallback):
          - scores: {ticker: TSI score in 0.0-1.0}.
          - is_neutral_fallback: {ticker: True} for any ticker that lacked
            enough price history (<30 bars) to compute a real TSI and was
            given the neutral 0.5 default instead. Previously this default
            was indistinguishable from a genuinely computed, middling TSI —
            see compute_composite_scores' data_quality flag for how this
            is surfaced to callers.
    """
    scores: Dict[str, float] = {}
    is_neutral_fallback: Dict[str, bool] = {}
    start_history = as_of - pd.Timedelta(days=100)

    for symbol in symbols:
        bars_qs = get_price_bars(symbol, start_history, as_of)
        df = dataframe_from_bars(bars_qs)
        if df.empty or len(df) < 30:
            scores[symbol.ticker] = 0.5
            is_neutral_fallback[symbol.ticker] = True
            logger.info(
                f"Technical score for {symbol.ticker} defaulted to neutral (0.5) — "
                f"only {len(df)} price bars available, need >=30."
            )
            continue

        close = df['Close']
        delta = close.diff()
        abs_delta = delta.abs()

        smooth1 = delta.ewm(span=25, adjust=False).mean()
        double_smooth = smooth1.ewm(span=13, adjust=False).mean()
        abs_smooth1 = abs_delta.ewm(span=25, adjust=False).mean()
        abs_double_smooth = abs_smooth1.ewm(span=13, adjust=False).mean()

        tsi = 100 * double_smooth / (abs_double_smooth + 1e-9)
        tsi_val = float(tsi.iloc[-1])

        # Normalize: (tanh(tsi/50) + 1) / 2 -> maps to 0-1
        scores[symbol.ticker] = float((np.tanh(tsi_val / 50.0) + 1) / 2)
        is_neutral_fallback[symbol.ticker] = False

    return scores, is_neutral_fallback


def compute_composite_scores(
    symbols: List[Symbol],
    as_of: date,
    transformer_scores: Dict[str, float],
    weights_config: Dict[str, float],
    per_model_scores: Optional[Dict[str, Dict[str, float]]] = None,
    contributing_models: Optional[Dict[str, List[str]]] = None,
    n_models_requested: Optional[int] = None,
) -> Dict[str, Dict]:
    """Blends Transformer, Quality, and Technical scores.

    Args:
        symbols: Symbols to score.
        as_of: Date to score as of.
        transformer_scores: {ticker: weighted ensemble score}, typically the
            first return value of forecasting.services.ensemble.run_ensemble_layer.
        weights_config: Layer weights, e.g. {"transformer": 0.8, "quality": 0.12,
            "technical": 0.08}.
        per_model_scores: {model_name: {ticker: raw_score}}, for per-model
            breakdown reporting.
        contributing_models: {ticker: [model_name, ...]} — the third return
            value of run_ensemble_layer. When supplied, any ticker scored by
            fewer than `n_models_requested` models is flagged in the
            returned dict's data_quality.partial_ensemble field, so a
            composite score built on 1-of-3 models is visibly distinct from
            one built on 3-of-3, instead of being an indistinguishable float.
        n_models_requested: Total number of models in the ensemble config.
            Required (and only meaningful) alongside contributing_models.

    Returns:
        {ticker: {..., "data_quality": {...}}} — see inline comments below
        for the exact fields. "no data" (couldn't be computed at all, fell
        back to a neutral default) is now distinguished from a genuinely
        computed neutral score, for both the quality and technical layers,
        and from partial ensemble coverage for the transformer layer. Every
        one of these conditions previously produced the same output shape
        as a fully-scored symbol, indistinguishable to any downstream
        consumer (persisted CompositeScore rows, Excel/PDF reports, or the
        ranking table) without re-deriving it from scratch.
    """
    logger.info("Fetching Quality scores from DB...")
    qual_qs = get_quality_scores_bulk(symbols, as_of)
    quality_scores = {q.symbol.ticker: q.composite_score for q in qual_qs}
    quality_covered = set(quality_scores.keys())

    logger.info("Computing Technical TSI scores from DB...")
    technical_scores, technical_is_neutral_fallback = get_technical_scores(symbols, as_of)

    # Normalize weights configuration
    active_w = {k: v for k, v in weights_config.items() if v > 0}
    total_w = sum(active_w.values())
    if total_w == 0:
        raise ValueError("All composite weights are zero.")
    norm_w = {k: v / total_w for k, v in active_w.items()}

    transformer_norm = _minmax_normalise(transformer_scores)

    low_quality_count = 0
    results = {}
    for sym in [s.ticker for s in symbols]:
        t_raw = transformer_scores.get(sym, 0.0)
        t_norm = transformer_norm.get(sym, 0.5)
        q = quality_scores.get(sym, 0.5)
        tc = technical_scores.get(sym, 0.5)

        composite = (
            norm_w.get("transformer", 0.0) * t_norm +
            norm_w.get("quality", 0.0) * q +
            norm_w.get("technical", 0.0) * tc
        )

        sym_per_model = {}
        if per_model_scores:
            for mname, mscores in per_model_scores.items():
                if sym in mscores:
                    sym_per_model[mname] = round(mscores[sym], 5)

        # ── Data quality flags ──────────────────────────────────────
        # quality_no_data: get_quality_scores_bulk had no QualityScore row
        # for this symbol at all (vs. a row whose composite_score happened
        # to equal ~0.5 through genuine computation).
        quality_no_data = sym not in quality_covered

        # technical_no_data: fewer than 30 price bars were available, so
        # get_technical_scores used its neutral fallback rather than a
        # real TSI computation.
        technical_no_data = technical_is_neutral_fallback.get(sym, True)

        # partial_ensemble: this symbol's transformer score was built from
        # fewer than the full requested model set (see run_ensemble_layer's
        # contributing_models return value). None when the caller didn't
        # supply ensemble coverage info (e.g. a single-model, non-ensemble
        # caller), since "partial" is meaningless without a total to compare against.
        n_contributing = None
        partial_ensemble = None
        if contributing_models is not None and n_models_requested:
            n_contributing = len(contributing_models.get(sym, []))
            partial_ensemble = n_contributing < n_models_requested

        is_low_quality = quality_no_data or technical_no_data or bool(partial_ensemble)
        if is_low_quality:
            low_quality_count += 1
            logger.info(
                f"{sym}: composite score has reduced data quality — "
                f"quality_no_data={quality_no_data}, technical_no_data={technical_no_data}, "
                f"partial_ensemble={partial_ensemble}"
                + (f" ({n_contributing}/{n_models_requested} models)" if n_models_requested else "")
            )

        results[sym] = {
            "composite_score": round(composite, 4),
            "transformer_score": round(t_raw, 5),
            "transformer_norm": round(t_norm, 4),
            "quality_score": round(q, 4),
            "technical_score": round(tc, 4),
            "weights_used": norm_w,
            "per_model_scores": sym_per_model,
            "data_quality": {
                "quality_no_data": quality_no_data,
                "technical_no_data": technical_no_data,
                "partial_ensemble": partial_ensemble,
                "n_contributing_models": n_contributing,
                "n_requested_models": n_models_requested,
            },
        }

    if low_quality_count:
        logger.warning(
            f"{low_quality_count} of {len(results)} symbols have a composite score built on "
            f"partial data (missing quality/technical history or partial ensemble coverage) — "
            f"see each symbol's 'data_quality' dict."
        )

    return results


def compute_config_hash(
    model_names: List[str],
    model_weights: List[float],
    weights_config: Dict[str, float],
    use_sentiment: bool,
) -> str:
    """Derives a stable short hash identifying a composite scoring configuration.

    Only inputs that change the *scores themselves* are included — model
    selection/weights, layer weights, and the sentiment flag. Portfolio
    construction params (universe, top_n, amount) are deliberately excluded:
    they change which stocks get bought, not how any given stock is scored,
    so two runs differing only in --top-n should share the same config_hash
    and update the same CompositeScore rows rather than fork into duplicates.
    """
    payload = {
        "models": sorted(zip(model_names, model_weights)),
        "weights_config": {k: round(v, 6) for k, v in sorted(weights_config.items())},
        "use_sentiment": use_sentiment,
    }
    raw = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(raw.encode()).hexdigest()[:16]


def save_composite_scores_to_db(
    composite_results: Dict[str, Dict],
    tiers: Dict[str, str],
    as_of_date: date,
    config_hash: str,
) -> QuerySet[CompositeScore]:
    """Persists composite scoring results as CompositeScore rows.

    Existing rows for this (as_of_date, config_hash) pair are cleared first,
    so re-running an identical config on the same date replaces rather than
    conflicting with (or duplicating) prior results — same idempotent-rerun
    pattern as portfolio_service.save_portfolio_to_db.
    """
    CompositeScore.objects.filter(as_of_date=as_of_date, config_hash=config_hash).delete()

    tickers = list(composite_results.keys())
    symbol_map = {s.ticker: s for s in Symbol.objects.filter(ticker__in=tickers)}

    rows_to_create = []
    for ticker, d in composite_results.items():
        symbol = symbol_map.get(ticker)
        if symbol is None:
            logger.warning(f"Symbol {ticker} not found in database. Skipping CompositeScore.")
            continue
        # data_quality is folded into the existing per_model_scores JSONField
        # under a reserved "_data_quality" key rather than adding a new
        # CompositeScore column — avoids a migration for what is, for now,
        # diagnostic metadata rather than a queried/filtered field. If this
        # needs to be filterable later (e.g. "show me only high-confidence
        # composite scores"), promote it to a real column then.
        per_model_payload = dict(d.get("per_model_scores", {}))
        per_model_payload["_data_quality"] = d.get("data_quality", {})

        rows_to_create.append(
            CompositeScore(
                symbol=symbol,
                as_of_date=as_of_date,
                config_hash=config_hash,
                composite_score=d["composite_score"],
                transformer_score=d["transformer_score"],
                transformer_norm=d["transformer_norm"],
                quality_score=d["quality_score"],
                technical_score=d["technical_score"],
                per_model_scores=per_model_payload,
                tier=tiers.get(ticker, "AVOID"),
            )
        )

    if rows_to_create:
        CompositeScore.objects.bulk_create(rows_to_create)

    return CompositeScore.objects.filter(as_of_date=as_of_date, config_hash=config_hash)


def _fetch_latest_price(ticker: str, as_of: date) -> float:
    """Returns the most recent DB-cached closing price for `ticker` on or before `as_of`.

    Replaces the previous fallback chain that attempted `from portfolio import
    fetch_latest_price` and `from portfolio_fixed import fetch_latest_price` —
    neither module exists anywhere in this codebase (there is a `portfolio`
    Django *app*, but no root-level `portfolio.py`/`portfolio_fixed.py` with
    that function), so both imports always failed and control always fell
    through to the inline fallback. That fallback was itself broken: it
    called `get_price_bars(s_obj)` with no `start`/`end` arguments, but
    `get_price_bars(symbol, start, end)` requires both — a guaranteed
    `TypeError` at runtime.

    This function is the single, correctly-implemented replacement: it
    queries `PriceBar` for a short window ending at `as_of` and returns the
    latest available close, or 0.0 if nothing is cached in that window
    (the caller already treats price<=0 as "unavailable").

    Args:
        ticker: The symbol's ticker (e.g. 'RELIANCE.NS').
        as_of: The date to price as of. The lookup window ends here and
            looks back PRICE_LOOKUP_WINDOW_DAYS days to tolerate short gaps.

    Returns:
        The latest close price on or before as_of, or 0.0 if none is cached.
    """
    try:
        symbol_obj = Symbol.objects.get(ticker=ticker)
    except Symbol.DoesNotExist:
        logger.warning(f"Symbol {ticker} not found in database; cannot price it.")
        return 0.0

    start = as_of - timedelta(days=PRICE_LOOKUP_WINDOW_DAYS)
    bars_qs = get_price_bars(symbol_obj, start, as_of)
    df = dataframe_from_bars(bars_qs)

    if df.empty:
        logger.warning(f"No cached price bars for {ticker} in [{start}, {as_of}].")
        return 0.0

    return float(df["Close"].iloc[-1])


def build_composite_portfolio(
    composite_results: dict,
    top_n: int,
    portfolio_amount: float,
    as_of: Optional[date] = None,
    weighting: str = "score",
) -> list:
    """
    Converts composite scores into a concrete portfolio with rupee allocations
    and share quantities.

    Args:
        composite_results: Per-symbol composite scoring output from
            compute_composite_scores.
        top_n: Number of BUY-tier symbols to allocate capital across.
        portfolio_amount: Total rupee capital to allocate.
        as_of: Date to price holdings as of. Defaults to today if omitted —
            pass the run's actual as_of_date explicitly where available so
            backtested/historical composite runs price against the correct
            date rather than "now".
        weighting: Weighting scheme passed through to construct_portfolio.
    """
    if not composite_results:
        return []

    from portfolio_optimizer import construct_portfolio

    pricing_date = as_of or date.today()

    comp_scores = {sym: d["composite_score"] for sym, d in composite_results.items()}
    weights_dict = construct_portfolio(comp_scores, top_n=top_n, weighting=weighting)

    today_str = datetime.now().strftime("%Y-%m-%d")
    holdings = []

    for sym, weight in weights_dict.items():
        alloc_rs = portfolio_amount * weight
        price = _fetch_latest_price(sym, pricing_date)
        qty = int(round(alloc_rs / price)) if price > 0 else 0

        detail = composite_results.get(sym, {})
        holdings.append({
            "symbol": sym,
            "quantity": qty,
            "purchase_price": round(price, 2),
            "price_source": "market_close" if price > 0 else "unavailable",
            "purchase_date": today_str,
            "allocation_pct": round(weight * 100, 2),
            "allocation_rs": round(alloc_rs, 2),
            "composite_score": detail.get("composite_score", 0.0),
            "transformer_score": detail.get("transformer_score", 0.0),
            "quality_score": detail.get("quality_score", 0.0),
            "technical_score": detail.get("technical_score", 0.0),
        })

    return holdings