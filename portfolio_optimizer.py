"""
portfolio_optimizer.py - Stock ranking, portfolio construction, backtesting

Functions:
  rank_stocks()          - sort stocks by predicted return score
  construct_portfolio()  - assign weights (equal / score-proportional)
  backtest_strategy()    - walk-forward backtest on historical predictions
"""

import warnings
import numpy as np
import pandas as pd
from typing import Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

from util import log, rank_normalize, METRICS_DIR, save_json

# ─────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────
RISK_FREE_RATE = 0.065   # 6.5% annualized (approx Indian T-bill)


# ─────────────────────────────────────────────
# Ranking
# ─────────────────────────────────────────────
def rank_stocks(
    predicted_scores: Dict[str, float],
    top_n:  int = 10,
    hold_n: int = 20,
) -> Dict[str, str]:
    """
    Rank stocks and assign BUY / HOLD / AVOID.

    Returns dict: {symbol: "BUY" | "HOLD" | "AVOID"}
    """
    if not predicted_scores:
        return {}

    series = pd.Series(predicted_scores).sort_values(ascending=False)
    tiers  = {}
    for i, (sym, score) in enumerate(series.items()):
        if i < top_n:
            tiers[sym] = "BUY"
        elif i < top_n + hold_n:
            tiers[sym] = "HOLD"
        else:
            tiers[sym] = "AVOID"
    return tiers


def score_to_ranking_table(
    predicted_scores: Dict[str, float],
) -> pd.DataFrame:
    """Return a sorted DataFrame with rank, score, percentile, tier."""
    series = pd.Series(predicted_scores).sort_values(ascending=False)
    df = pd.DataFrame({
        "symbol": series.index,
        "raw_score": series.values,
    })
    df["percentile"]   = df["raw_score"].rank(pct=True).round(3)
    df["rank"]         = range(1, len(df) + 1)
    df["norm_score"]   = rank_normalize(df["raw_score"]).round(4)
    return df.set_index("symbol")


# ─────────────────────────────────────────────
# Portfolio construction
# ─────────────────────────────────────────────
def construct_portfolio(
    predicted_scores: Dict[str, float],
    top_n:     int   = 10,
    weighting: str   = "score",   # "equal" | "score" | "sqrt_score"
    max_weight: float = 0.20,
) -> Dict[str, float]:
    """
    Build a long-only portfolio from top_n predicted scores.

    Returns dict: {symbol: weight}
    """
    series = pd.Series(predicted_scores).sort_values(ascending=False).head(top_n)

    # Keep only positive scores
    series = series[series > 0]
    if series.empty:
        log.warning("No positive predicted scores; equal-weighting all candidates.")
        series = pd.Series(predicted_scores).sort_values(ascending=False).head(top_n)
        series[:] = 1.0

    if weighting == "equal":
        weights = pd.Series(1.0 / len(series), index=series.index)
    elif weighting == "score":
        weights = series / series.sum()
    elif weighting == "sqrt_score":
        sq = np.sqrt(series.clip(lower=0))
        weights = sq / sq.sum()
    else:
        weights = pd.Series(1.0 / len(series), index=series.index)

    # Cap individual weights
    weights = weights.clip(upper=max_weight)
    weights = weights / weights.sum()   # renormalize after cap

    return weights.round(4).to_dict()


# ─────────────────────────────────────────────
# Performance metrics
# ─────────────────────────────────────────────
def sharpe_ratio(returns: pd.Series, freq: int = 252) -> float:
    if returns.std() == 0:
        return 0.0
    excess = returns.mean() - RISK_FREE_RATE / freq
    return float((excess / returns.std()) * np.sqrt(freq))


def sortino_ratio(returns: pd.Series, freq: int = 252) -> float:
    down = returns[returns < 0]
    if down.std() == 0:
        return 0.0
    excess = returns.mean() - RISK_FREE_RATE / freq
    return float((excess / down.std()) * np.sqrt(freq))


def max_drawdown(equity: pd.Series) -> float:
    roll_max = equity.cummax()
    drawdown = (equity - roll_max) / roll_max
    return float(drawdown.min())


def information_ratio(
    portfolio_returns: pd.Series,
    benchmark_returns: pd.Series,
) -> float:
    active = portfolio_returns - benchmark_returns
    if active.std() == 0:
        return 0.0
    return float(active.mean() / active.std() * np.sqrt(252))


def portfolio_metrics(
    equity_curve: pd.Series,
    benchmark_curve: Optional[pd.Series] = None,
) -> dict:
    returns = equity_curve.pct_change().dropna()
    metrics = {
        "total_return":    float((equity_curve.iloc[-1] / equity_curve.iloc[0]) - 1),
        "cagr":            float(((equity_curve.iloc[-1] / equity_curve.iloc[0])
                                  ** (252 / max(len(equity_curve), 1)) - 1)),
        "sharpe":          sharpe_ratio(returns),
        "sortino":         sortino_ratio(returns),
        "max_drawdown":    max_drawdown(equity_curve),
        "volatility_ann":  float(returns.std() * np.sqrt(252)),
        "hit_ratio":       float((returns > 0).mean()),
    }
    if benchmark_curve is not None:
        bench_rets = benchmark_curve.pct_change().dropna()
        metrics["information_ratio"] = information_ratio(returns, bench_rets)
        metrics["benchmark_return"]  = float(
            (benchmark_curve.iloc[-1] / benchmark_curve.iloc[0]) - 1)
    return metrics


# ─────────────────────────────────────────────
# Backtesting
# ─────────────────────────────────────────────
def backtest_strategy(
    all_close_prices: Dict[str, pd.Series],   # symbol → Close series
    predicted_scores_by_date: Dict[str, Dict[str, float]],  # date_str → {sym: score}
    rebalance_freq: int  = 21,   # trading days between rebalance
    top_n:          int  = 10,
    weighting:      str  = "score",
    benchmark_sym:  str  = "^NSEI",
    benchmark_close: Optional[pd.Series] = None,
    initial_capital: float = 1_000_000.0,
) -> Tuple[pd.Series, dict]:
    """
    Walk-forward portfolio backtest.

    Parameters
    ----------
    all_close_prices : dict of Close price series per symbol
    predicted_scores_by_date : dict where key=date string,
                               value=dict of {symbol: pred_score}
    rebalance_freq : days between portfolio rebalances
    top_n          : number of longs
    weighting      : "equal" | "score"

    Returns
    -------
    equity_curve : pd.Series
    metrics      : dict
    """
    if not predicted_scores_by_date:
        log.warning("No predictions provided; backtest skipped.")
        return pd.Series(dtype=float), {}

    # Build a combined price DataFrame
    price_df = pd.DataFrame(all_close_prices).sort_index().fillna(method="ffill")
    common_dates = price_df.index

    # Daily returns for each stock
    daily_ret = price_df.pct_change().fillna(0)

    equity      = initial_capital
    equity_hist = []
    date_hist   = []
    current_weights: Dict[str, float] = {}
    rebal_counter = 0

    sorted_dates = sorted(predicted_scores_by_date.keys())

    for i, dt in enumerate(common_dates):
        dt_str = str(dt.date())

        # Rebalance on schedule or when new predictions available
        if rebal_counter == 0 and dt_str in predicted_scores_by_date:
            scores  = predicted_scores_by_date[dt_str]
            current_weights = construct_portfolio(scores, top_n=top_n, weighting=weighting)
            rebal_counter   = rebalance_freq

        # Apply daily returns
        port_ret = 0.0
        for sym, w in current_weights.items():
            if sym in daily_ret.columns and dt in daily_ret.index:
                port_ret += w * daily_ret.loc[dt, sym]

        equity *= (1 + port_ret)
        equity_hist.append(equity)
        date_hist.append(dt)

        if rebal_counter > 0:
            rebal_counter -= 1

    equity_curve = pd.Series(equity_hist, index=date_hist)

    # Benchmark
    bench_series = None
    if benchmark_close is not None and not benchmark_close.empty:
        bench = benchmark_close.reindex(common_dates).ffill().dropna()
        bench_series = bench / bench.iloc[0] * initial_capital

    metrics = portfolio_metrics(equity_curve, bench_series)
    log.info(f"Backtest complete | Return={metrics['total_return']:.2%}  "
             f"Sharpe={metrics['sharpe']:.2f}  MaxDD={metrics['max_drawdown']:.2%}")

    # Persist
    save_json(metrics, METRICS_DIR / "backtest_metrics.json")
    return equity_curve, metrics


# ─────────────────────────────────────────────
# Walk-forward validation split helper
# ─────────────────────────────────────────────
def walk_forward_splits(
    n: int,
    train_ratio: float = 0.70,
    val_ratio:   float = 0.15,
) -> List[Tuple[slice, slice, slice]]:
    """
    Returns list of (train_slice, val_slice, test_slice) index slices
    for a single walk-forward fold.
    """
    train_end = int(n * train_ratio)
    val_end   = int(n * (train_ratio + val_ratio))
    return [(slice(0, train_end),
             slice(train_end, val_end),
             slice(val_end, n))]


def rolling_walk_forward_splits(
    n:              int,
    train_size:     int,
    val_size:       int,
    test_size:      int,
    step:           int = None,
) -> List[Tuple[slice, slice, slice]]:
    """
    Expanding-window walk-forward splits.
    """
    step = step or test_size
    splits = []
    start  = 0
    while start + train_size + val_size + test_size <= n:
        t_end = start + train_size
        v_end = t_end + val_size
        s_end = v_end + test_size
        splits.append((slice(start, t_end), slice(t_end, v_end), slice(v_end, s_end)))
        start += step
    return splits


# Default weights — configurable at call site
HYBRID_WEIGHTS = {
    "transformer": 0.40,
    "factor":      0.25,
    "indicator":   0.20,
    "sentiment":   0.15,
}

def compute_hybrid_score(
    transformer_score: float,
    factor_score:      float = 0.0,
    indicator_score:   float = 0.0,
    sentiment_score:   float = 0.0,
    weights:           dict  = None,
) -> float:
    """
    Combine model outputs into one final score.

    FinalScore =
        0.40 * TransformerPrediction
      + 0.25 * FactorScore
      + 0.20 * IndicatorScore
      + 0.15 * SentimentScore

    All inputs should be normalised to the same scale before calling
    (e.g. z-scored across the universe on prediction day).
    Weights are configurable via the weights dict.
    """
    w = weights or HYBRID_WEIGHTS
    return (
        w.get("transformer", 0.40) * transformer_score +
        w.get("factor",      0.25) * factor_score      +
        w.get("indicator",   0.20) * indicator_score   +
        w.get("sentiment",   0.15) * sentiment_score
    )


def rank_stocks_hybrid(
    transformer_scores: dict,
    factor_scores:      dict = None,
    indicator_scores:   dict = None,
    sentiment_scores:   dict = None,
    weights:            dict = None,
    top_n:              int  = 10,
    hold_n:             int  = 15,
) -> dict:
    """
    Drop-in replacement for rank_stocks() that uses hybrid scoring.
    Any score dict not supplied defaults to 0 contribution.
    Returns tier dict: {symbol: 'BUY' | 'HOLD' | 'AVOID'}
    """
    all_syms = set(transformer_scores.keys())
    hybrid = {}
    for sym in all_syms:
        hybrid[sym] = compute_hybrid_score(
            transformer_score = transformer_scores.get(sym, 0.0),
            factor_score      = (factor_scores     or {}).get(sym, 0.0),
            indicator_score   = (indicator_scores  or {}).get(sym, 0.0),
            sentiment_score   = (sentiment_scores  or {}).get(sym, 0.0),
            weights           = weights,
        )
    return rank_stocks(hybrid, top_n=top_n, hold_n=hold_n)