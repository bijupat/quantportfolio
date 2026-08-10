"""
financial_indicators.py
Advanced quantitative indicators beyond the core technical set.
Sits alongside feature_engineering.py and feeds into it.

Categories:
  - Momentum:        ROC, Williams %R, CCI, TSI
  - Volatility:      Historical, Parkinson, Garman-Klass
  - Market Breadth:  Advance/Decline, New Highs/Lows, Breadth Index
  - Relative Strength: vs index, vs sector
"""

import numpy as np
import pandas as pd
from typing import Optional
from util import log


# ══════════════════════════════════════════
# MOMENTUM INDICATORS
# ══════════════════════════════════════════

def calculate_roc(close: pd.Series, period: int = 10) -> pd.Series:
    """
    Rate of Change — percentage price change over N periods.
    ROC = (close_today - close_N_days_ago) / close_N_days_ago * 100
    Positive = upward momentum. Negative = downward momentum.
    """
    roc = close.pct_change(period) * 100
    roc.name = f"roc_{period}"
    return roc


def calculate_williams_r(
    high: pd.Series,
    low:  pd.Series,
    close: pd.Series,
    period: int = 14,
) -> pd.Series:
    """
    Williams %R — measures overbought/oversold on -100 to 0 scale.
    -80 to -100 = oversold (potential buy)
     -0 to -20  = overbought (potential sell)
    Uses highest high and lowest low over the lookback window.
    """
    highest_high = high.rolling(period).max()
    lowest_low   = low.rolling(period).min()
    wr = -100 * (highest_high - close) / (highest_high - lowest_low + 1e-9)
    wr.name = f"williams_r_{period}"
    return wr


def calculate_cci(
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Commodity Channel Index — measures deviation from average price.
    > +100 = potential overbought / breakout
    < -100 = potential oversold / breakdown
    Typical price = (H + L + C) / 3
    CCI = (typical - SMA(typical)) / (0.015 * mean_abs_deviation)
    """
    typical = (high + low + close) / 3
    sma_tp  = typical.rolling(period).mean()
    mad     = typical.rolling(period).apply(
        lambda x: np.abs(x - x.mean()).mean(), raw=True
    )
    cci = (typical - sma_tp) / (0.015 * mad + 1e-9)
    cci.name = f"cci_{period}"
    return cci


def calculate_tsi(
    close:        pd.Series,
    slow_period:  int = 25,
    fast_period:  int = 13,
) -> pd.Series:
    """
    True Strength Index — double-smoothed momentum oscillator.
    Ranges roughly -100 to +100.
    Positive and rising = strong uptrend.
    Negative and falling = strong downtrend.
    Less whipsaw than RSI due to double smoothing.
    """
    delta      = close.diff()
    abs_delta  = delta.abs()

    # Double exponential smoothing
    smooth1       = delta.ewm(span=slow_period, adjust=False).mean()
    double_smooth = smooth1.ewm(span=fast_period, adjust=False).mean()

    abs_smooth1       = abs_delta.ewm(span=slow_period, adjust=False).mean()
    abs_double_smooth = abs_smooth1.ewm(span=fast_period, adjust=False).mean()

    tsi = 100 * double_smooth / (abs_double_smooth + 1e-9)
    tsi.name = f"tsi_{slow_period}_{fast_period}"
    return tsi


# ══════════════════════════════════════════
# VOLATILITY INDICATORS
# ══════════════════════════════════════════

def calculate_historical_volatility(
    close:  pd.Series,
    period: int = 20,
    annualise: bool = True,
) -> pd.Series:
    """
    Historical (realised) volatility — standard deviation of log returns.
    Annualised by multiplying by sqrt(252).
    High HV = uncertain/trending market.
    Low HV  = calm market (often precedes a big move).
    """
    log_ret = np.log(close / close.shift(1))
    hv = log_ret.rolling(period).std()
    if annualise:
        hv = hv * np.sqrt(252)
    hv.name = f"hist_vol_{period}"
    return hv


def calculate_parkinson_volatility(
    high:   pd.Series,
    low:    pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Parkinson Volatility — uses high/low range instead of close-to-close.
    More efficient estimator than historical vol — captures intraday moves.
    Parkinson = sqrt( (1/(4*N*ln2)) * sum(ln(H/L)^2) ) * sqrt(252)
    """
    log_hl = np.log(high / low) ** 2
    pv = np.sqrt(log_hl.rolling(period).mean() / (4 * np.log(2))) * np.sqrt(252)
    pv.name = f"parkinson_vol_{period}"
    return pv


def calculate_garman_klass_volatility(
    open_:  pd.Series,
    high:   pd.Series,
    low:    pd.Series,
    close:  pd.Series,
    period: int = 20,
) -> pd.Series:
    """
    Garman-Klass Volatility — most efficient OHLC volatility estimator.
    Incorporates open, close, high, low for maximum information.
    GK = sqrt( 252/N * sum(
        0.5*(ln(H/L))^2 - (2*ln2-1)*(ln(C/O))^2
    ))
    Better than Parkinson because it uses close vs open return too.
    """
    log_hl = 0.5 * np.log(high / low) ** 2
    log_co = (2 * np.log(2) - 1) * np.log(close / open_) ** 2
    gk     = np.sqrt(252 * (log_hl - log_co).rolling(period).mean())
    gk.name = f"garman_klass_vol_{period}"
    return gk


# ══════════════════════════════════════════
# MARKET BREADTH INDICATORS
# (index-level: NIFTY50 / SENSEX components)
# ══════════════════════════════════════════

def advance_decline_ratio(prices_df: pd.DataFrame) -> pd.Series:
    """
    Advance/Decline Ratio — fraction of stocks rising on each day.
    prices_df: DataFrame where each column is one stock's Close price.
    > 0.6 = broad advance (healthy rally)
    < 0.4 = broad decline (weak market)
    Values near 0.5 = mixed market
    """
    daily_ret = prices_df.pct_change()
    advances  = (daily_ret > 0).sum(axis=1)
    declines  = (daily_ret < 0).sum(axis=1)
    adr = advances / (declines + 1e-9)
    adr.name = "advance_decline_ratio"
    return adr


def new_highs_new_lows(
    prices_df: pd.DataFrame,
    period:    int = 52,
) -> pd.DataFrame:
    """
    New Highs / New Lows — counts stocks at N-week highs vs lows.
    Returns DataFrame with columns: new_highs, new_lows, hl_ratio.
    hl_ratio > 1 = more stocks making new highs = bullish breadth.
    period=52 approximates 1-year (52 * 5 trading days).
    """
    window = period * 5   # convert weeks to approx trading days
    new_highs = (prices_df >= prices_df.rolling(window).max().shift(1)).sum(axis=1)
    new_lows  = (prices_df <= prices_df.rolling(window).min().shift(1)).sum(axis=1)
    return pd.DataFrame({
        "new_highs": new_highs,
        "new_lows":  new_lows,
        "hl_ratio":  new_highs / (new_lows + 1e-9),
    })


def market_breadth_index(
    prices_df: pd.DataFrame,
    sma_period: int = 200,
) -> pd.Series:
    """
    Market Breadth Index — fraction of stocks above their N-day SMA.
    > 0.7 = strong breadth (most stocks in uptrend)
    < 0.3 = weak breadth (most stocks below long-term average)
    Used as a market regime filter: only take BUY signals when breadth > 0.5.
    """
    above_sma = (
        prices_df > prices_df.rolling(sma_period).mean()
    ).sum(axis=1) / prices_df.shape[1]
    above_sma.name = f"breadth_pct_above_sma{sma_period}"
    return above_sma


# ══════════════════════════════════════════
# RELATIVE STRENGTH INDICATORS
# ══════════════════════════════════════════

def relative_strength_vs_index(
    stock_close: pd.Series,
    index_close: pd.Series,
    period:      int = 63,   # ~1 quarter
) -> pd.Series:
    """
    Relative Strength vs Index — measures if stock is outperforming.
    RS = (stock_return_N_days) / (index_return_N_days)
    > 1.0 = stock outperforming index
    < 1.0 = stock underperforming index
    Normalised: (RS - 1) so 0 = in line with index.
    """
    stock_ret = stock_close.pct_change(period)
    index_ret = index_close.pct_change(period)
    rs = (stock_ret - index_ret)   # excess return vs benchmark
    rs.name = f"rs_vs_index_{period}d"
    return rs


def relative_strength_vs_sector(
    stock_close:  pd.Series,
    sector_closes: pd.DataFrame,
    period:        int = 63,
) -> pd.Series:
    """
    Relative Strength vs Sector — measures if stock leads its sector.
    sector_closes: DataFrame of peer stocks in the same sector.
    Returns stock excess return minus median peer return.
    Positive = stock is a sector leader.
    """
    stock_ret  = stock_close.pct_change(period)
    sector_med = sector_closes.pct_change(period).median(axis=1)
    rs_sector  = stock_ret - sector_med
    rs_sector.name = f"rs_vs_sector_{period}d"
    return rs_sector


# ══════════════════════════════════════════
# MASTER BUILDER
# Called from feature_engineering.py
# ══════════════════════════════════════════

def build_advanced_indicators(
    ohlcv:        pd.DataFrame,
    index_close:  Optional[pd.Series] = None,
    sector_closes: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Compute all advanced indicators for a single stock OHLCV DataFrame.
    Merges results into a single DataFrame aligned to ohlcv.index.
    Safe — any indicator that fails is skipped with a warning.

    Args:
        ohlcv:         DataFrame with columns Open, High, Low, Close, Volume
        index_close:   NIFTY/SENSEX Close series for relative strength
        sector_closes: DataFrame of sector peer Close prices
    Returns:
        DataFrame with one column per indicator, aligned to ohlcv.index
    """
    o = ohlcv["Open"]
    h = ohlcv["High"]
    l = ohlcv["Low"]
    c = ohlcv["Close"]

    results = {}

    # ── Momentum ──────────────────────────────────────────────
    for fn, kwargs in [
        (calculate_roc,       {"close": c, "period": 10}),
        (calculate_roc,       {"close": c, "period": 20}),
        (calculate_williams_r,{"high": h, "low": l, "close": c, "period": 14}),
        (calculate_cci,       {"high": h, "low": l, "close": c, "period": 20}),
        (calculate_tsi,       {"close": c}),
    ]:
        try:
            s = fn(**kwargs)
            results[s.name] = s
        except Exception as e:
            log.warning(f"Indicator {fn.__name__} failed: {e}")

    # ── Volatility ────────────────────────────────────────────
    for fn, kwargs in [
        (calculate_historical_volatility, {"close": c, "period": 20}),
        (calculate_historical_volatility, {"close": c, "period": 60}),
        (calculate_parkinson_volatility,  {"high": h, "low": l, "period": 20}),
        (calculate_garman_klass_volatility,{"open_": o, "high": h, "low": l, "close": c, "period": 20}),
    ]:
        try:
            s = fn(**kwargs)
            results[s.name] = s
        except Exception as e:
            log.warning(f"Indicator {fn.__name__} failed: {e}")

    # ── Relative Strength (optional) ──────────────────────────
    if index_close is not None:
        for period in [21, 63]:
            try:
                s = relative_strength_vs_index(c, index_close, period)
                results[s.name] = s
            except Exception as e:
                log.warning(f"relative_strength_vs_index({period}) failed: {e}")

    if sector_closes is not None:
        try:
            s = relative_strength_vs_sector(c, sector_closes)
            results[s.name] = s
        except Exception as e:
            log.warning(f"relative_strength_vs_sector failed: {e}")

    out = pd.DataFrame(results, index=ohlcv.index)
    log.info(f"  Advanced indicators built: {len(out.columns)} columns")
    return out