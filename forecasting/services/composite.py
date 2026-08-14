import hashlib
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional

from django.db.models import QuerySet

from core.models import Symbol
from market_data.services.prices import get_price_bars, dataframe_from_bars
from market_data.services.quality import get_quality_scores_bulk
from forecasting.models import CompositeScore

logger = logging.getLogger(__name__)

def _minmax_normalise(scores: Dict[str, float]) -> Dict[str, float]:
    """Min-max normalise a score dict to 0.0-1.0."""
    if not scores: return {}
    vals = np.array(list(scores.values()), dtype=float)
    lo, hi = vals.min(), vals.max()
    if hi == lo: return {sym: 0.5 for sym in scores}
    return {sym: float((v - lo) / (hi - lo)) for sym, v in scores.items()}

def get_technical_scores(symbols: List[Symbol], as_of: date) -> Dict[str, float]:
    """Computes TSI (True Strength Index) from DB PriceBars and maps to 0.0-1.0."""
    scores = {}
    start_history = as_of - pd.Timedelta(days=100)
    
    for symbol in symbols:
        bars_qs = get_price_bars(symbol, start_history, as_of)
        df = dataframe_from_bars(bars_qs)
        if df.empty or len(df) < 30:
            scores[symbol.ticker] = 0.5
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
        
    return scores

def compute_composite_scores(
    symbols: List[Symbol],
    as_of: date,
    transformer_scores: Dict[str, float],
    weights_config: Dict[str, float],
    per_model_scores: Optional[Dict[str, Dict[str, float]]] = None
) -> Dict[str, Dict]:
    """Blends Transformer, Quality, and Technical scores."""
    logger.info("Fetching Quality scores from DB...")
    qual_qs = get_quality_scores_bulk(symbols, as_of)
    quality_scores = {q.symbol.ticker: q.composite_score for q in qual_qs}
    
    logger.info("Computing Technical TSI scores from DB...")
    technical_scores = get_technical_scores(symbols, as_of)
    
    # Normalize weights configuration
    active_w = {k: v for k, v in weights_config.items() if v > 0}
    total_w = sum(active_w.values())
    if total_w == 0:
        raise ValueError("All composite weights are zero.")
    norm_w = {k: v / total_w for k, v in active_w.items()}
    
    transformer_norm = _minmax_normalise(transformer_scores)
    
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
                    
        results[sym] = {
            "composite_score": round(composite, 4),
            "transformer_score": round(t_raw, 5),
            "transformer_norm": round(t_norm, 4),
            "quality_score": round(q, 4),
            "technical_score": round(tc, 4),
            "weights_used": norm_w,
            "per_model_scores": sym_per_model
        }
        
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
                per_model_scores=d.get("per_model_scores", {}),
                tier=tiers.get(ticker, "AVOID"),
            )
        )

    if rows_to_create:
        CompositeScore.objects.bulk_create(rows_to_create)

    return CompositeScore.objects.filter(as_of_date=as_of_date, config_hash=config_hash)

def build_composite_portfolio(
    composite_results: dict,
    top_n: int,
    portfolio_amount: float,
    weighting: str = "score",
) -> list:
    """
    Converts composite scores into a concrete portfolio with rupee allocations
    and share quantities.
    """
    if not composite_results:
        return []

    from portfolio_optimizer import construct_portfolio
    
    # Try importing fetch_latest_price from portfolio.py, with a fallback if needed
    try:
        from portfolio import fetch_latest_price
    except ImportError:
        try:
            from portfolio_fixed import fetch_latest_price
        except ImportError:
            # Fallback inline price fetcher using price history if portfolio.py isn't found
            def fetch_latest_price(sym):
                from market_data.services.prices import get_price_bars, dataframe_from_bars
                from core.models import Symbol
                try:
                    s_obj = Symbol.objects.get(ticker=sym)
                    bars = get_price_bars(s_obj).order_by('-date')[:5]
                    df = dataframe_from_bars(bars)
                    if not df.empty:
                        return float(df['Close'].iloc[-1])
                except Exception:
                    pass
                return 0.0

    comp_scores = {sym: d["composite_score"] for sym, d in composite_results.items()}
    weights_dict = construct_portfolio(comp_scores, top_n=top_n, weighting=weighting)

    today_str = datetime.now().strftime("%Y-%m-%d")
    holdings = []

    for sym, weight in weights_dict.items():
        alloc_rs = portfolio_amount * weight
        price = fetch_latest_price(sym)
        if price is None or price <= 0:
            price = 0.0
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