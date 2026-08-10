import logging
from datetime import date
from django.db import transaction
from django.db.models import QuerySet
import pandas as pd

from core.models import Symbol
from market_data.models import QualityScore

try:
    from altman import calculate_altman_z_score
    from piotroski import calculate_piotroski_score
except ImportError:
    pass

logger = logging.getLogger(__name__)

def compute_composite_quality(piotroski_score: int, altman_z: float, excluded: bool) -> float:
    """Replicates the 0.0 to 1.0 composite blend from quality_factor.py"""
    p_norm = piotroski_score / 9.0 if piotroski_score is not None else 0.5
    
    if excluded or altman_z is None:
        return p_norm  # 100% weight to Piotroski if Altman is invalid
        
    # Cap Altman at 4.0 for normalization purposes
    a_norm = min(max(altman_z, 0.0), 4.0) / 4.0
    return (p_norm * 0.6) + (a_norm * 0.4)

def get_quality_scores_bulk(symbols: list[Symbol], as_of: date) -> QuerySet[QualityScore]:
    """
    Bulk DB-first fetcher for fundamental quality scores.
    """
    # 1. Find which symbols already have a quality score for this date
    symbol_ids = [s.id for s in symbols]
    existing = QualityScore.objects.filter(symbol_id__in=symbol_ids, date=as_of)
    existing_ids = set(existing.values_list("symbol_id", flat=True))
    
    missing_symbols = [s for s in symbols if s.id not in existing_ids]
    
    if missing_symbols:
        logger.info(f"Computing fundamental quality scores for {len(missing_symbols)} symbols")
        
        scores_to_create = []
        
        for symbol in missing_symbols:
            try:
                # Run Original Scripts
                p_result = calculate_piotroski_score(symbol.ticker)
                a_result = calculate_altman_z_score(symbol.ticker)
                
                p_score = p_result.get("piotroski_score")
                a_score = a_result.get("z_score")
                excluded = a_result.get("excluded", False)
                
                composite = compute_composite_quality(p_score, a_score, excluded)
                
                scores_to_create.append(
                    QualityScore(
                        symbol=symbol,
                        date=as_of,
                        piotroski_score=p_score,
                        altman_z=a_score,
                        excluded_sector=excluded,
                        composite_score=composite
                    )
                )
            except Exception as e:
                logger.error(f"Failed to calculate quality score for {symbol.ticker}: {e}")
        
        # Save to DB
        with transaction.atomic():
            QualityScore.objects.bulk_create(scores_to_create, ignore_conflicts=True)
            
    return QualityScore.objects.filter(symbol_id__in=symbol_ids, date=as_of)