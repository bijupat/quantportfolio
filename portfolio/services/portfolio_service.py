"""Services for persisting portfolios: DB writes and legacy CSV import.

Both entry points funnel through save_portfolio_to_db(), which upserts a
Portfolio row and replaces its PortfolioItem rows — see that function's
docstring for the allow_empty guard added to prevent a bad/empty holdings
list from silently wiping a previously-good portfolio.
"""

import logging
from typing import Dict, List, Optional

import pandas as pd
from django.db import transaction

from core.models import Symbol
from portfolio.models import Portfolio, PortfolioItem, PortfolioStrategy

logger = logging.getLogger(__name__)

# Columns load_portfolio_from_csv_to_db requires to be present (not
# necessarily fully populated) before it will attempt an import. Missing
# any of these means the CSV isn't a portfolio export at all, and failing
# fast here is clearer than the confusing downstream errors/NaNs that
# resulted from a column silently defaulting to 0 for every row.
REQUIRED_CSV_COLUMNS = ("symbol", "quantity", "purchase_price")


@transaction.atomic
def save_portfolio_to_db(
    name: str,
    holdings: List[Dict],
    total_amount: float,
    strategy_tag: str = PortfolioStrategy.MANUAL,
    owner=None,
    allow_empty: bool = False,
) -> Portfolio:
    """Saves a portfolio and its line items into the database.

    Args:
        name: Unique Portfolio.name. An existing portfolio with this name
            is updated in place (its old items are replaced).
        holdings: List of dicts with keys: symbol, quantity, purchase_price,
            allocation_pct, allocation_rs. See load_portfolio_from_csv_to_db
            and forecasting.services.composite.build_composite_portfolio for
            the two current producers of this shape. Callers are responsible
            for ensuring quantity/purchase_price/allocation_rs are >= 0 and
            allocation_pct is within [0, 100] — see PortfolioItem.Meta.constraints
            in portfolio/models.py, which reject out-of-range values at the
            database level with django.db.IntegrityError. This function does
            not pre-validate signs itself; load_portfolio_from_csv_to_db does,
            since it's the only current producer that takes untrusted external
            numeric input (build_composite_portfolio's own arithmetic can't
            produce negative values — see that function's docstring).
        total_amount: Total rupee capital represented by holdings.
        strategy_tag: One of PortfolioStrategy's values, identifying which
            pipeline produced this portfolio (used by portfolio.views to
            decide cross-user visibility — see PortfolioStrategy.COMPOSITE_AI).
        owner: The core.User this portfolio belongs to, if any. None produces
            a system-generated/ownerless portfolio (e.g. CLI/legacy imports).
        allow_empty: Must be explicitly True to save/update a portfolio with
            an empty holdings list. Without this, calling save_portfolio_to_db
            with holdings=[] against an EXISTING portfolio name raises
            ValueError instead of silently deleting every PortfolioItem the
            portfolio previously had — e.g. because an upstream ranking step
            selected zero BUY-tier stocks, or a malformed CSV parsed to zero
            valid rows. Creating a brand-new portfolio with no holdings is
            still allowed either way, since there's nothing to lose there.

    Returns:
        The created/updated Portfolio instance.

    Raises:
        ValueError: If holdings is empty, an existing portfolio with this
            name already has items, and allow_empty is not True.
        django.db.IntegrityError: If any holding violates a PortfolioItem
            database constraint (negative quantity/price/allocation_rs, or
            allocation_pct outside [0, 100]) — see portfolio/models.py.
            Callers that accept untrusted input (e.g. CSV rows) should
            validate before calling this function rather than relying on
            this as the only backstop, since one bad row can also skew
            OTHER rows' derived allocation_pct if it isn't caught upstream.
    """
    existing = Portfolio.objects.filter(name=name).first()
    if not holdings and existing is not None and existing.items.exists() and not allow_empty:
        raise ValueError(
            f"Refusing to save portfolio '{name}' with an empty holdings list — "
            f"it currently has {existing.items.count()} item(s). This would "
            f"silently delete all of them. Pass allow_empty=True if that's "
            f"actually intended."
        )

    portfolio, _ = Portfolio.objects.update_or_create(
        name=name,
        defaults={
            "total_capital": total_amount,
            "strategy": strategy_tag,
            "is_active": True,
            "owner": owner,
        },
    )
    # Clear old items if updating an existing active portfolio
    portfolio.items.all().delete()

    items_to_create = []
    for h in holdings:
        ticker = h.get("symbol")
        try:
            symbol_obj = Symbol.objects.get(ticker=ticker)
        except Symbol.DoesNotExist:
            logger.warning(f"Symbol {ticker} not found in database. Skipping item.")
            continue

        items_to_create.append(
            PortfolioItem(
                portfolio=portfolio,
                symbol=symbol_obj,
                quantity=h.get("quantity", 0),
                purchase_price=h.get("purchase_price", 0.0),
                allocation_pct=h.get("allocation_pct", 0.0),
                allocation_rs=h.get("allocation_rs", 0.0),
            )
        )

    if items_to_create:
        PortfolioItem.objects.bulk_create(items_to_create)

    logger.info(f"Successfully saved portfolio '{name}' with {len(items_to_create)} items to DB.")
    return portfolio


def _coerce_numeric_column(df: pd.DataFrame, column: str) -> pd.Series:
    """Coerces a DataFrame column to numeric, turning unparsable values into NaN.

    Distinguishes "column missing entirely" (caller's responsibility to check
    beforehand via REQUIRED_CSV_COLUMNS) from "column present but contains a
    blank/non-numeric cell" — the latter is what silently produced NaN
    quantities/prices before this fix, since Series.get(key, default) only
    ever falls back to `default` when `key` itself is absent, never when the
    cell value is NaN or a stray string.
    """
    return pd.to_numeric(df[column], errors="coerce")


def load_portfolio_from_csv_to_db(csv_path: str, portfolio_name: str) -> Portfolio:
    """Imports a legacy portfolio CSV into the database.

    Expects at minimum a `symbol`, `quantity`, and `purchase_price` column.
    Rows with a missing symbol, a non-numeric quantity/purchase_price, or a
    NEGATIVE quantity/purchase_price are skipped (and counted) rather than
    silently coerced or passed through. Two distinct bugs are guarded against
    here:

      1. A blank "purchase_price" cell previously became NaN via
         DataFrame.get(), which then propagated into total_val, every row's
         allocation_pct (NaN / NaN), and finally into the saved
         PortfolioItem rows with no error raised anywhere in the pipeline.
         Fixed by pd.to_numeric(errors="coerce") + a notna() filter.

      2. A NEGATIVE quantity or purchase_price parses successfully (it's a
         valid number, just the wrong sign) and passes the notna() filter
         above untouched. Without an explicit sign check here, such a row
         reaches PortfolioItem.objects.bulk_create() in save_portfolio_to_db,
         where it either raises an unhandled django.db.IntegrityError against
         the quantity/purchase_price/allocation_rs CheckConstraints (see
         portfolio/models.py) — or, worse, silently corrupts every OTHER
         valid row's allocation_pct: allocation_rs = quantity * purchase_price
         is included in total_val before any constraint is checked, so one
         large negative row skews the denominator every row's percentage is
         computed against. Sign validation must happen here, at row-selection
         time, before total_val is ever summed — catching it only at the DB
         layer is too late to protect the other rows in the same import.

    Args:
        csv_path: Path to the CSV file to import.
        portfolio_name: Name to save the resulting Portfolio under. An
            existing portfolio with this name is updated in place.

    Returns:
        The created/updated Portfolio instance.

    Raises:
        ValueError: If the CSV is missing a required column, or if every
            row fails validation (leaving nothing importable).
    """
    df = pd.read_csv(csv_path)

    missing_columns = [c for c in REQUIRED_CSV_COLUMNS if c not in df.columns]
    if missing_columns:
        raise ValueError(
            f"CSV '{csv_path}' is missing required column(s): {', '.join(missing_columns)}. "
            f"Expected at least: {', '.join(REQUIRED_CSV_COLUMNS)}."
        )

    df["quantity"] = _coerce_numeric_column(df, "quantity")
    df["purchase_price"] = _coerce_numeric_column(df, "purchase_price")

    # A row is importable only if: it has a non-blank symbol, both numeric
    # fields parsed successfully, AND both numeric fields are non-negative.
    # Rows failing any of these are dropped up front so total_val (and
    # therefore every row's allocation_pct) is computed only from genuinely
    # valid, non-negative rows — a single negative row must not be allowed
    # to skew the shared denominator that every other row's percentage
    # divides by.
    valid_symbol = df["symbol"].notna() & (df["symbol"].astype(str).str.strip() != "")
    valid_quantity = df["quantity"].notna() & (df["quantity"] >= 0)
    valid_price = df["purchase_price"].notna() & (df["purchase_price"] >= 0)
    is_valid_row = valid_symbol & valid_quantity & valid_price

    n_total = len(df)
    n_invalid = int((~is_valid_row).sum())
    if n_invalid:
        invalid_symbols = df.loc[~is_valid_row, "symbol"].fillna("<blank>").astype(str).tolist()
        logger.warning(
            f"Skipping {n_invalid} of {n_total} row(s) in '{csv_path}' with a missing symbol, "
            f"non-numeric quantity/purchase_price, or a negative quantity/purchase_price: "
            f"{invalid_symbols}"
        )

    df = df[is_valid_row].copy()
    if df.empty:
        raise ValueError(
            f"CSV '{csv_path}' had no valid rows after validation "
            f"({n_invalid} of {n_total} row(s) were skipped). Nothing to import."
        )

    holdings = []
    total_val = 0.0

    for _, row in df.iterrows():
        val = float(row["quantity"]) * float(row["purchase_price"])
        total_val += val
        holdings.append({
            "symbol": row["symbol"],
            "quantity": int(row["quantity"]),
            "purchase_price": float(row["purchase_price"]),
            "allocation_pct": 0.0,  # normalised below
            "allocation_rs": val,
        })

    # total_val is guaranteed >= 0 here: every row's quantity and
    # purchase_price passed the >= 0 filter above, so their product (and the
    # running sum) can never go negative. Still guard the degenerate
    # all-zero-value case (e.g. every row legitimately has purchase_price=0)
    # rather than dividing by zero.
    for h in holdings:
        h["allocation_pct"] = (h["allocation_rs"] / total_val * 100) if total_val > 0 else 0.0

    return save_portfolio_to_db(
        portfolio_name,
        holdings,
        total_val,
        strategy_tag=PortfolioStrategy.LEGACY_CSV,
        allow_empty=False,
    )