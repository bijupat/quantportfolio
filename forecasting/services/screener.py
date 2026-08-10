import logging
import pandas as pd
from datetime import date, timedelta
from django.db import transaction
from django.db.models import QuerySet

from core.models import Symbol
from market_data.models import PriceBar
from market_data.services.prices import dataframe_from_bars, get_price_bars
from forecasting.models import ScreenerResult

logger = logging.getLogger(__name__)

def _compute_signals(df: pd.DataFrame) -> dict:
    """Computes the 6 technical screening signals from OHLCV data."""
    if df is None or len(df) < 65:
        return {}

    close = df["Close"]
    volume = df["Volume"]
    high = df["High"]

    last_close = float(close.iloc[-1])

    # S1: Trend
    sma20 = float(close.rolling(20).mean().iloc[-1])
    sma50 = float(close.rolling(50).mean().iloc[-1])
    s1 = bool(last_close > sma20 and last_close > sma50)

    # S2: Momentum
    mom_20d = float(close.pct_change(20).iloc[-1]) if len(close) > 20 else 0.0
    mom_60d = float(close.pct_change(60).iloc[-1]) if len(close) > 60 else 0.0
    s2 = bool(mom_20d > 0 and mom_60d > 0)

    # S3: MACD
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    macd_hist = float((macd_line - macd_signal).iloc[-1])
    s3 = bool(macd_hist >= 0)

    # S4: RSI
    delta = close.diff()
    gain = delta.clip(lower=0).rolling(14).mean()
    loss = (-delta.clip(upper=0)).rolling(14).mean()
    rsi = float((100 - 100 / (1 + gain / (loss + 1e-9))).iloc[-1])
    s4 = bool(35 <= rsi <= 72)

    # S5: Volume
    vol_sma20 = float(volume.rolling(20).mean().iloc[-1])
    vol_ratio = float(volume.iloc[-1]) / (vol_sma20 + 1e-9)
    s5 = bool(vol_ratio >= 1.0)

    # S6: Drawdown
    high_252 = float(high.rolling(252).max().iloc[-1]) if len(high) >= 252 else float(high.max())
    drawdown = (last_close - high_252) / (high_252 + 1e-9)
    s6 = bool(drawdown >= -0.15)

    score = sum([s1, s2, s3, s4, s5, s6])

    return {
        "s1_trend": s1, "s2_momentum": s2, "s3_macd": s3,
        "s4_rsi": s4, "s5_volume": s5, "s6_drawdown": s6,
        "score": score,
        "raw_values": {
            "close": last_close, "mom_20d": mom_20d, "mom_60d": mom_60d,
            "macd_hist": macd_hist, "rsi_14": rsi, "vol_ratio": vol_ratio,
            "drawdown": drawdown
        }
    }

def get_or_screen(symbols: list[Symbol], as_of: date, min_score: int = 3, top_n: int = 30) -> QuerySet[ScreenerResult]:
    """DB-first screener cache layer."""
    symbol_ids = [s.id for s in symbols]
    
    existing_qs = ScreenerResult.objects.filter(symbol_id__in=symbol_ids, as_of_date=as_of)
    existing_ids = set(existing_qs.values_list("symbol_id", flat=True))
    
    missing_symbols = [s for s in symbols if s.id not in existing_ids]
    
    if missing_symbols:
        logger.info(f"Calculating screening signals for {len(missing_symbols)} symbols...")
        results_to_create = []
        
        # Need ~300 days of history for 52-week rolling high
        start_history = as_of - timedelta(days=300)
        
        for symbol in missing_symbols:
            bars_qs = get_price_bars(symbol, start_history, as_of)
            df = dataframe_from_bars(bars_qs)
            
            signals = _compute_signals(df)
            
            if signals:
                results_to_create.append(
                    ScreenerResult(
                        symbol=symbol, as_of_date=as_of,
                        s1_trend=signals["s1_trend"], s2_momentum=signals["s2_momentum"],
                        s3_macd=signals["s3_macd"], s4_rsi=signals["s4_rsi"],
                        s5_volume=signals["s5_volume"], s6_drawdown=signals["s6_drawdown"],
                        score=signals["score"], status="PENDING", raw_values=signals["raw_values"]
                    )
                )
            else:
                results_to_create.append(
                    ScreenerResult(
                        symbol=symbol, as_of_date=as_of,
                        s1_trend=False, s2_momentum=False, s3_macd=False, 
                        s4_rsi=False, s5_volume=False, s6_drawdown=False,
                        score=0, status="NO DATA", raw_values={}
                    )
                )
                
        with transaction.atomic():
            ScreenerResult.objects.bulk_create(results_to_create, ignore_conflicts=True)

    # Dynamic Status Labeling (since min_score and top_n can change per run)
    final_qs = ScreenerResult.objects.filter(symbol_id__in=symbol_ids, as_of_date=as_of)
    valid_results = sorted([r for r in final_qs if r.status != "NO DATA"], key=lambda x: (-x.score, -(x.raw_values.get('mom_60d', 0))))
    
    updates = []
    for rank, result in enumerate(valid_results):
        if result.score >= min_score:
            new_status = "PASS" if rank < top_n else "PASS (not selected)"
        elif result.score == min_score - 1:
            new_status = "NEAR MISS"
        else:
            new_status = "FAIL"
            
        if result.status != new_status:
            result.status = new_status
            updates.append(result)
            
    if updates:
        ScreenerResult.objects.bulk_update(updates, ['status'])

    return final_qs