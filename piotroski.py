"""
piotroski.py — Piotroski F-Score calculation for NSE/BSE stocks

The Piotroski F-Score is a 9-point scoring system that measures
corporate financial health across three dimensions:

  Profitability   (4 signals)
  Leverage        (3 signals)
  Efficiency      (2 signals)

Each signal scores 1 (positive) or 0 (negative).
Total score 0-9:
  7-9 = Strong financial health
  4-6 = Average
  0-3 = Weak / potential distress

Developed by Joseph Piotroski (2000) — adapted here for Indian GAAP.

Usage:
    from piotroski import calculate_piotroski_score
    result = calculate_piotroski_score("RELIANCE.NS")
    print(result["piotroski_score"])   # e.g. 7
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
    log = logging.getLogger("piotroski")
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s  %(levelname)-7s  %(message)s")

# ─────────────────────────────────────────────
# Financial sector tickers — Piotroski still
# valid for these but interpret with caution
# (Altman Z-Score excludes these entirely)
# ─────────────────────────────────────────────
FINANCIAL_SECTOR_KEYWORDS = [
    "BANK", "NBFC", "FINANCE", "FINANCIAL",
    "INSURANCE", "HDFC", "ICICI", "AXIS",
    "KOTAK", "INDUSIND", "BAJAJFIN",
]


def _is_financial_stock(symbol: str) -> bool:
    """Heuristic check — flag for informational warning only."""
    upper = symbol.upper()
    return any(kw in upper for kw in FINANCIAL_SECTOR_KEYWORDS)


def _safe_get(data: dict, *keys, default=None):
    """
    Safely retrieve a value from nested dict trying multiple key aliases.
    yfinance field names differ between versions and between stocks.
    Returns first non-None match, else default.
    """
    for key in keys:
        val = data.get(key)
        if val is not None and not (isinstance(val, float) and np.isnan(val)):
            return val
    return default


def _fetch_financials(symbol: str) -> Optional[Dict]:
    """
    Fetch all required financial statement data for Piotroski calculation.
    Returns a flat dict with standardised keys, or None on complete failure.

    yfinance returns:
        ticker.info            — snapshot metrics
        ticker.financials      — annual income statement (rows=metrics, cols=dates)
        ticker.balance_sheet   — annual balance sheet
        ticker.cashflow        — annual cash flow statement
    """
    try:
        ticker = yf.Ticker(symbol)

        # ── Income statement ──────────────────────────────────
        fin = ticker.financials
        # ── Balance sheet ────────────────────────────────────
        bs  = ticker.balance_sheet
        # ── Cash flow ─────────────────────────────────────────
        cf  = ticker.cashflow
        # ── Info snapshot ─────────────────────────────────────
        info = ticker.info or {}

        def _row(df, *keys):
            """Get most recent and prior year value for a statement row."""
            if df is None or df.empty:
                return None, None
            for key in keys:
                matches = [k for k in df.index if key.lower() in k.lower()]
                if matches:
                    row = df.loc[matches[0]]
                    vals = [v for v in row.values
                            if v is not None and not (isinstance(v, float) and np.isnan(v))]
                    curr = vals[0] if len(vals) > 0 else None
                    prev = vals[1] if len(vals) > 1 else None
                    return curr, prev
            return None, None

        # ── Extract values ─────────────────────────────────────
        net_income_curr, net_income_prev   = _row(fin,
            "Net Income", "NetIncome", "Net Income Common Stockholders")
        op_cashflow_curr, op_cashflow_prev = _row(cf,
            "Operating Cash Flow", "Total Cash From Operating Activities",
            "Cash From Operations")
        total_assets_curr, total_assets_prev = _row(bs,
            "Total Assets", "TotalAssets")
        long_term_debt_curr, long_term_debt_prev = _row(bs,
            "Long Term Debt", "LongTermDebt", "Long-Term Debt")
        current_assets_curr, current_assets_prev = _row(bs,
            "Current Assets", "Total Current Assets", "CurrentAssets")
        current_liab_curr, current_liab_prev     = _row(bs,
            "Current Liabilities", "Total Current Liabilities", "CurrentLiabilities")
        shares_curr, shares_prev = _row(bs,
            "Common Stock", "Ordinary Shares Number", "Share Issued")
        gross_profit_curr, gross_profit_prev = _row(fin,
            "Gross Profit", "GrossProfit")
        revenue_curr, revenue_prev = _row(fin,
            "Total Revenue", "Revenue", "TotalRevenue")
        ebit_curr, _ = _row(fin,
            "EBIT", "Operating Income", "Earnings Before Interest And Taxes")

        data = {
            "symbol":               symbol,
            "net_income_curr":      net_income_curr,
            "net_income_prev":      net_income_prev,
            "op_cashflow_curr":     op_cashflow_curr,
            "total_assets_curr":    total_assets_curr,
            "total_assets_prev":    total_assets_prev,
            "long_term_debt_curr":  long_term_debt_curr,
            "long_term_debt_prev":  long_term_debt_prev,
            "current_assets_curr":  current_assets_curr,
            "current_assets_prev":  current_assets_prev,
            "current_liab_curr":    current_liab_curr,
            "current_liab_prev":    current_liab_prev,
            "shares_curr":          shares_curr,
            "shares_prev":          shares_prev,
            "gross_profit_curr":    gross_profit_curr,
            "gross_profit_prev":    gross_profit_prev,
            "revenue_curr":         revenue_curr,
            "revenue_prev":         revenue_prev,
            "ebit":                 ebit_curr,
            "market_cap":           _safe_get(info, "marketCap"),
            "total_liabilities":    _safe_get(info, "totalDebt"),
        }
        return data

    except Exception as e:
        log.error(f"  [{symbol}] Failed to fetch financials: {e}")
        return None


def _roa(net_income, total_assets) -> Optional[float]:
    """Return on Assets = Net Income / Total Assets"""
    if net_income is None or total_assets is None or total_assets == 0:
        return None
    return net_income / total_assets


def _current_ratio(current_assets, current_liab) -> Optional[float]:
    if current_assets is None or current_liab is None or current_liab == 0:
        return None
    return current_assets / current_liab


def _gross_margin(gross_profit, revenue) -> Optional[float]:
    if gross_profit is None or revenue is None or revenue == 0:
        return None
    return gross_profit / revenue


def _asset_turnover(revenue, total_assets) -> Optional[float]:
    if revenue is None or total_assets is None or total_assets == 0:
        return None
    return revenue / total_assets


# ─────────────────────────────────────────────
# MAIN FUNCTION
# ─────────────────────────────────────────────

def calculate_piotroski_score(symbol: str) -> Dict:
    """
    Calculate the 9-point Piotroski F-Score for a given stock.

    Args:
        symbol: Yahoo Finance ticker (e.g. 'RELIANCE.NS')

    Returns:
        dict with keys:
            symbol              — ticker
            piotroski_score     — integer 0-9
            signals             — dict of all 9 individual signal scores
            warnings            — list of data quality warnings
            data_quality        — fraction of signals with full data (0.0-1.0)
    """
    result = {
        "symbol":          symbol,
        "piotroski_score": None,
        "signals":         {},
        "warnings":        [],
        "data_quality":    0.0,
    }

    if _is_financial_stock(symbol):
        result["warnings"].append(
            f"Financial sector stock detected ({symbol}). "
            "Piotroski is applicable but interpret leverage signals with caution "
            "as banking balance sheets differ from industrial companies."
        )

    log.info(f"  [{symbol}] Fetching financial data for Piotroski…")
    data = _fetch_financials(symbol)

    if data is None:
        result["warnings"].append("Could not fetch financial data — score unavailable.")
        return result

    signals = {}
    missing = []

    # ══════════════════════════════════════════
    # PROFITABILITY (4 signals)
    # ══════════════════════════════════════════

    # F1 — Positive Net Income
    if data["net_income_curr"] is not None:
        signals["F1_positive_net_income"] = 1 if data["net_income_curr"] > 0 else 0
    else:
        signals["F1_positive_net_income"] = 0
        missing.append("net_income")

    # F2 — Positive Operating Cash Flow
    if data["op_cashflow_curr"] is not None:
        signals["F2_positive_op_cashflow"] = 1 if data["op_cashflow_curr"] > 0 else 0
    else:
        signals["F2_positive_op_cashflow"] = 0
        missing.append("op_cashflow")

    # F3 — Increasing ROA (ROA this year > ROA last year)
    roa_curr = _roa(data["net_income_curr"], data["total_assets_curr"])
    roa_prev = _roa(data["net_income_prev"], data["total_assets_prev"])
    if roa_curr is not None and roa_prev is not None:
        signals["F3_increasing_roa"] = 1 if roa_curr > roa_prev else 0
    else:
        signals["F3_increasing_roa"] = 0
        missing.append("roa_comparison")

    # F4 — Accruals: Operating Cash Flow > Net Income (quality of earnings)
    if data["op_cashflow_curr"] is not None and data["net_income_curr"] is not None \
            and data["total_assets_curr"] is not None and data["total_assets_curr"] != 0:
        accrual = (data["op_cashflow_curr"] - data["net_income_curr"]) \
                  / data["total_assets_curr"]
        signals["F4_low_accruals"] = 1 if accrual > 0 else 0
    else:
        signals["F4_low_accruals"] = 0
        missing.append("accruals")

    # ══════════════════════════════════════════
    # LEVERAGE & LIQUIDITY (3 signals)
    # ══════════════════════════════════════════

    # F5 — Decreasing Long-Term Debt ratio
    lev_curr = (data["long_term_debt_curr"] / data["total_assets_curr"]
                if data["long_term_debt_curr"] is not None
                and data["total_assets_curr"] not in (None, 0) else None)
    lev_prev = (data["long_term_debt_prev"] / data["total_assets_prev"]
                if data["long_term_debt_prev"] is not None
                and data["total_assets_prev"] not in (None, 0) else None)
    if lev_curr is not None and lev_prev is not None:
        signals["F5_lower_leverage"] = 1 if lev_curr < lev_prev else 0
    else:
        signals["F5_lower_leverage"] = 0
        missing.append("leverage_comparison")

    # F6 — Increasing Current Ratio (liquidity improving)
    cr_curr = _current_ratio(data["current_assets_curr"], data["current_liab_curr"])
    cr_prev = _current_ratio(data["current_assets_prev"], data["current_liab_prev"])
    if cr_curr is not None and cr_prev is not None:
        signals["F6_higher_current_ratio"] = 1 if cr_curr > cr_prev else 0
    else:
        signals["F6_higher_current_ratio"] = 0
        missing.append("current_ratio_comparison")

    # F7 — No New Share Issuance (dilution check)
    if data["shares_curr"] is not None and data["shares_prev"] is not None:
        signals["F7_no_dilution"] = 1 if data["shares_curr"] <= data["shares_prev"] else 0
    else:
        # Default to 1 (benefit of doubt) if data unavailable
        signals["F7_no_dilution"] = 1
        missing.append("shares_outstanding")

    # ══════════════════════════════════════════
    # EFFICIENCY (2 signals)
    # ══════════════════════════════════════════

    # F8 — Increasing Gross Margin
    gm_curr = _gross_margin(data["gross_profit_curr"], data["revenue_curr"])
    gm_prev = _gross_margin(data["gross_profit_prev"], data["revenue_prev"])
    if gm_curr is not None and gm_prev is not None:
        signals["F8_higher_gross_margin"] = 1 if gm_curr > gm_prev else 0
    else:
        signals["F8_higher_gross_margin"] = 0
        missing.append("gross_margin_comparison")

    # F9 — Increasing Asset Turnover (efficiency improving)
    at_curr = _asset_turnover(data["revenue_curr"], data["total_assets_curr"])
    at_prev = _asset_turnover(data["revenue_prev"], data["total_assets_prev"])
    if at_curr is not None and at_prev is not None:
        signals["F9_higher_asset_turnover"] = 1 if at_curr > at_prev else 0
    else:
        signals["F9_higher_asset_turnover"] = 0
        missing.append("asset_turnover_comparison")

    # ── Total score ───────────────────────────────────────────
    total = sum(signals.values())
    filled = 9 - len(missing)
    data_quality = filled / 9

    if missing:
        result["warnings"].append(
            f"Missing data for {len(missing)} signal(s): {', '.join(missing)}. "
            "These defaulted to 0. Score may be understated."
        )

    result["piotroski_score"] = total
    result["signals"]         = signals
    result["data_quality"]    = round(data_quality, 2)

    interpretation = (
        "Strong"  if total >= 7 else
        "Average" if total >= 4 else
        "Weak"
    )
    result["interpretation"] = interpretation

    log.info(
        f"  [{symbol}] Piotroski F-Score = {total}/9  "
        f"({interpretation})  data_quality={data_quality:.0%}"
    )
    return result


# ── CLI convenience ───────────────────────────────────────────
if __name__ == "__main__":
    import sys, json
    sym = sys.argv[1] if len(sys.argv) > 1 else "RELIANCE.NS"
    res = calculate_piotroski_score(sym)
    print(json.dumps(res, indent=2, default=str))
