"""
util.py - Logging, helpers, and shared utilities
"""

import os
import json
import logging
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

# ─────────────────────────────────────────────
# Directory Setup
# ─────────────────────────────────────────────
BASE_DIR      = Path(__file__).parent
MODELS_DIR    = BASE_DIR / "models"
METRICS_DIR   = BASE_DIR / "metrics"
DATA_CACHE    = BASE_DIR / "data_cache"
NEWS_CACHE    = BASE_DIR / "news_cache"

for d in [MODELS_DIR, METRICS_DIR, DATA_CACHE, NEWS_CACHE]:
    d.mkdir(parents=True, exist_ok=True)


# ─────────────────────────────────────────────
# Logger
# ─────────────────────────────────────────────
def get_logger(name: str = "QuantAI") -> logging.Logger:
    logger = logging.getLogger(name)
    if not logger.handlers:
        logger.setLevel(logging.DEBUG)
        fmt = logging.Formatter(
            "[%(asctime)s] %(levelname)-8s %(name)s | %(message)s",
            datefmt="%Y-%m-%d %H:%M:%S",
        )
        # Console
        ch = logging.StreamHandler()
        ch.setLevel(logging.INFO)
        ch.setFormatter(fmt)
        logger.addHandler(ch)
        # File
        fh = logging.FileHandler(BASE_DIR / "quant_ai.log")
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fmt)
        logger.addHandler(fh)
    return logger


log = get_logger()


# ─────────────────────────────────────────────
# JSON helpers
# ─────────────────────────────────────────────
def save_json(obj: dict, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, default=str)
    log.debug(f"Saved JSON -> {path}")


def load_json(path: Path) -> dict:
    with open(path, "r") as f:
        return json.load(f)


# ─────────────────────────────────────────────
# Numpy / DataFrame helpers
# ─────────────────────────────────────────────
def safe_divide(a, b, fill=0.0):
    with np.errstate(divide="ignore", invalid="ignore"):
        result = np.where(b != 0, a / b, fill)
    return result


def clip_outliers(series: pd.Series, n_std: float = 4.0) -> pd.Series:
    mean, std = series.mean(), series.std()
    return series.clip(mean - n_std * std, mean + n_std * std)


def normalize_min_max(series: pd.Series) -> pd.Series:
    mn, mx = series.min(), series.max()
    if mx == mn:
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - mn) / (mx - mn)


def rank_normalize(scores: pd.Series) -> pd.Series:
    """Convert raw scores to 0-1 percentile rank."""
    return scores.rank(pct=True)


# ─────────────────────────────────────────────
# Date helpers
# ─────────────────────────────────────────────
def trading_days_ago(n: int) -> str:
    """Return date string n calendar days ago (approximate for trading days)."""
    from datetime import timedelta
    return (datetime.today() - timedelta(days=int(n * 1.5))).strftime("%Y-%m-%d")


def today_str() -> str:
    return datetime.today().strftime("%Y-%m-%d")


# ─────────────────────────────────────────────
# Cache helpers
# ─────────────────────────────────────────────
def cache_path(symbol: str, prefix: str = "stock") -> Path:
    safe = symbol.replace(".", "_").replace("^", "IDX_")
    return DATA_CACHE / f"{prefix}_{safe}.parquet"


def news_cache_path(symbol: str) -> Path:
    safe = symbol.replace(".", "_").replace("^", "IDX_")
    return NEWS_CACHE / f"news_{safe}.json"


# ─────────────────────────────────────────────
# Sequence builder
# ─────────────────────────────────────────────
def build_sequences(
    data: np.ndarray,
    labels: np.ndarray,
    seq_len: int = 60,
) -> tuple:
    """
    Slide a window of length seq_len over data to produce
    (X, y) arrays suitable for sequence models.
    """
    X, y = [], []
    for i in range(len(data) - seq_len):
        X.append(data[i : i + seq_len])
        y.append(labels[i + seq_len])
    return np.array(X, dtype=np.float32), np.array(y, dtype=np.float32)


# ─────────────────────────────────────────────
# Metric formatters
# ─────────────────────────────────────────────
def format_metrics(metrics: dict) -> str:
    lines = ["-" * 50, "  Performance Metrics", "-" * 50]
    for k, v in metrics.items():
        if isinstance(v, float):
            lines.append(f"  {k:<30} {v:>10.4f}")
        else:
            lines.append(f"  {k:<30} {str(v):>10}")
    lines.append("-"  * 50)
    return "\n".join(lines)
