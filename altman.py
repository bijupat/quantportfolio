"""
altman.py — Altman Z-Score for NSE/BSE non-financial stocks

The Altman Z-Score predicts corporate bankruptcy probability
using five financial ratios:

  Z = 1.2*X1 + 1.4*X2 + 3.3*X3 + 0.6*X4 + 1.0*X5

  X1 = Working Capital / Total Assets
  X2 = Retained Earnings / Total Assets
  X3 = EBIT / Total Assets
  X4 = Market Value of Equity / Total Liabilities
  X5 = Revenue / Total Assets

Interpretation:
  Z > 3.0      Safe zone
  1.8–3.0      Grey zone (monitor closely)
  Z < 1.8      Distress zone

⚠️ IMPORTANT: Altman Z-Score is NOT valid for:
  - Banks and NBFCs
  - Insurance companies
  - Any financial sector stock

This module automatically detects and excludes financial stocks.
Developed by Edward Altman (1968) — adapted for Indian markets.

Usage:
    from altman import calculate_altman_z_score
    result = calculate_altman_z_score("RELIANCE.NS")
    print(result["z_score"])          # e.g. 3.42
    print(result["interpretation"])   # e.g. "Safe"
"""

import warnings
import numpy as np
import yfinance as yf
from typing import Dict, Optional

warnings.filterwarnings("ignore")

try:
    from util import log
except ImportError:
    import logging
    log = logging.getLogger("altman")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")

# ─────────────────────────────────────────────
# Financial sector exclusion list
# Altman Z-Score is structurally invalid for
# these business types — exclude entirely.
# ─────────────────────────────────────────────
FINANCIAL_SECTOR_PATTERNS = [
    "BANK", "HDFC", "ICICI", "AXIS", "KOTAK",
    "INDUSIND", "FEDERALBNK", "BANDHANBNK",
    "IDFCFIRST", "SBIN", "PNB", "BOB",
    "BAJAJFIN", "BAJAJHLDNG", "MUTHOOTFIN",
    "CHOLAFIN", "MANAPPURAM", "LICHSGFIN",
    "HDFCLIFE", "SBILIFE", "ICICIPRULI",
    "GICRE", "NIACL", "INSURANCE", "NBFC",
]

# Altman Z-Score weights (original Altman 1968)
W1, W2, W3, W4, W5 = 1.2, 1.4, 3.3, 0.6, 1.0


def _is_financial_sector(symbol: str) -> bool:
    """
    Returns True if the symbol is a financial sector stock.
    Altman Z-Score must not be applied to these.
    """
    upper = symbol.upper().replace(".NS", "").replace(".BO", "")
    return any(pattern in upper for pattern in FINANCIAL_SECTOR_PATTERNS)


def _row_value(df, *keys) -> Optional[float]:
    """
    Extract the most recent value from a financial statement DataFrame
    by trying multiple possible row name aliases.
    Handles yfinance column name inconsistencies across versions.
    """
    if df is None or df.empty:
        return None
    for key in keys:
        matches = [k for k in df.index if key.lower() in k.lower()]
        if matches:
            row = df.loc[matches[0]]
            vals = [v for v in row.values
                    if v is not None and not (isinstance(v, float) and np.isnan(v))]
            if vals:
                return float(vals[0])
    return None


def _fetch_altman_data(symbol: str) -> Optional[Dict]:
    """
    Fetch all five components needed for Altman Z-Score from yfinance.

    Components:
        working_capital     = Current Assets - Current Liabilities
        retained_earnings   = from balance sheet
        ebit                = Operating Income / EBIT from income statement
        market_cap          = from ticker.info
        total_liabilities   = Total Liabilities from balance sheet
        revenue             = Total Revenue from income statement
        total_assets        = Total Assets from balance sheet
    """
    try:
        ticker = yf.Ticker(symbol)
        info   = ticker.info or {}
        fin    = ticker.financials
        bs     = ticker.balance_sheet
        cf     = ticker.cashflow

        # ── Balance sheet items ───────────────────────────────
        total_assets    = _row_value(bs, "Total Assets", "TotalAssets")
        current_assets  = _row_value(bs, "Current Assets", "Total Current Assets")
        current_liab    = _row_value(bs, "Current Liabilities",
                                     "Total Current Liabilities")
        retained_earn   = _row_value(bs, "Retained Earnings",
                                     "Retained Earnings Total Equity",
                                     "RetainedEarnings")
        total_liab      = _row_value(bs, "Total Liabilities Net Minority Interest",
                                     "Total Liabilities", "TotalLiabilities",
                                     "Total Debt")

        # ── Income statement items ────────────────────────────
        ebit    = _row_value(fin, "EBIT", "Operating Income",
                              "Earnings Before Interest And Taxes",
                              "Operating Income Loss")
        revenue = _row_value(fin, "Total Revenue", "Revenue", "TotalRevenue")

        # ── Market data ───────────────────────────────────────
        market_cap = info.get("marketCap")

        # Working capital = current assets - current liabilities
        working_capital = None
        if current_assets is not None and current_liab is not None:
            working_capital = current_assets - current_liab

        # Fall back to totalDebt if total_liab unavailable
        if total_liab is None:
            total_liab = info.get("totalDebt")

        return {
            "symbol":           symbol,
            "total_assets":     total_assets,
            "working_capital":  working_capital,
            "retained_earnings":retained_earn,
            "ebit":             ebit,
            "market_cap":       market_cap,
            "total_liabilities":total_liab,
            "revenue":          revenue,
        }

    except Exception as e:
        log.error(f"  [{symbol}] Failed to fetch Altman data: {e}")
        return None


def calculate_altman_z_score(symbol: str) -> Dict:
    """
    Calculate Altman Z-Score for a given non-financial stock.

    Args:
        symbol: Yahoo Finance ticker (e.g. 'RELIANCE.NS')

    Returns:
        dict with keys:
            symbol          — ticker
            z_score         — float (None if excluded or insufficient data)
            components      — dict of X1..X5 values
            interpretation  — 'Safe' | 'Grey Zone' | 'Distress' | 'Excluded' | 'Insufficient Data'
            warnings        — list of data quality warnings
            excluded        — True if financial sector stock (Z-Score not applicable)
    """
    result = {
        "symbol":         symbol,
        "z_score":        None,
        "components":     {},
        "interpretation": None,
        "warnings":       [],
        "excluded":       False,
    }

    # ── Financial sector exclusion ────────────────────────────
    if _is_financial_sector(symbol):
        result["excluded"]       = True
        result["interpretation"] = "Excluded"
        result["warnings"].append(
            f"Altman Z-Score NOT applicable to financial sector stock ({symbol}). "
            "The model's assumptions about asset structure and liabilities do not hold "
            "for banks, NBFCs, and insurance companies."
        )
        log.warning(f"  [{symbol}] Excluded from Altman Z-Score — financial sector")
        return result

    log.info(f"  [{symbol}] Fetching financial data for Altman Z-Score…")
    data = _fetch_altman_data(symbol)

    if data is None:
        result["interpretation"] = "Insufficient Data"
        result["warnings"].append("Could not fetch financial data.")
        return result

    ta = data["total_assets"]
    if ta is None or ta == 0:
        result["interpretation"] = "Insufficient Data"
        result["warnings"].append(
            "Total Assets unavailable or zero — cannot compute Z-Score."
        )
        return result

    missing_fields = []

    # ── X1: Working Capital / Total Assets ────────────────────
    # Measures short-term liquidity relative to asset base.
    # Negative = current liabilities exceed current assets (liquidity stress).
    if data["working_capital"] is not None:
        X1 = data["working_capital"] / ta
    else:
        X1 = 0.0
        missing_fields.append("working_capital")

    # ── X2: Retained Earnings / Total Assets ──────────────────
    # Measures cumulative profitability. Younger companies score lower
    # even if currently profitable. Negative = accumulated losses.
    if data["retained_earnings"] is not None:
        X2 = data["retained_earnings"] / ta
    else:
        X2 = 0.0
        missing_fields.append("retained_earnings")

    # ── X3: EBIT / Total Assets ────────────────────────────────
    # Measures operating efficiency independent of taxes and leverage.
    # Most important component — highest coefficient (3.3).
    if data["ebit"] is not None:
        X3 = data["ebit"] / ta
    else:
        X3 = 0.0
        missing_fields.append("ebit")

    # ── X4: Market Value of Equity / Total Liabilities ─────────
    # Measures how much the firm's market value can decline before
    # liabilities exceed assets. Uses market cap as equity value.
    if data["market_cap"] is not None and data["total_liabilities"] not in (None, 0):
        X4 = data["market_cap"] / data["total_liabilities"]
    else:
        X4 = 0.0
        missing_fields.append("market_cap_or_liabilities")

    # ── X5: Revenue / Total Assets ────────────────────────────
    # Asset utilisation efficiency — how well assets generate sales.
    if data["revenue"] is not None:
        X5 = data["revenue"] / ta
    else:
        X5 = 0.0
        missing_fields.append("revenue")

    # ── Z-Score calculation ───────────────────────────────────
    z = W1*X1 + W2*X2 + W3*X3 + W4*X4 + W5*X5

    interpretation = (
        "Safe"       if z > 3.0  else
        "Grey Zone"  if z >= 1.8 else
        "Distress"
    )

    if missing_fields:
        result["warnings"].append(
            f"Missing data for: {', '.join(missing_fields)}. "
            "These defaulted to 0 — Z-Score may be understated."
        )

    result["z_score"]       = round(z, 4)
    result["components"]    = {
        "X1_working_capital_ratio":    round(X1, 4),
        "X2_retained_earnings_ratio":  round(X2, 4),
        "X3_ebit_ratio":               round(X3, 4),
        "X4_market_to_liabilities":    round(X4, 4),
        "X5_asset_turnover":           round(X5, 4),
    }
    result["interpretation"]   = interpretation
    result["data_completeness"] = round((5 - len(missing_fields)) / 5, 2)

    log.info(
        f"  [{symbol}] Altman Z-Score = {z:.2f}  ({interpretation})  "
        f"data={result['data_completeness']:.0%}"
    )
    return result


# ── CLI convenience ───────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    res = calculate_altman_z_score(sym)
    print(json.dumps(res, indent=2, default=str))
