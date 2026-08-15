import logging
import pandas as pd
from django.db import transaction
from typing import List, Dict

from core.models import Symbol
from portfolio.models import Portfolio, PortfolioItem

logger = logging.getLogger(__name__)

@transaction.atomic
def save_portfolio_to_db(
    name: str,
    holdings: List[Dict],
    total_amount: float,
    strategy_tag: str = "composite",
    owner=None,
) -> Portfolio:
    """
    Saves a portfolio and its line items into the database.

    Args:
        owner: The core.User this portfolio belongs to, if any. None produces
            a system-generated/ownerless portfolio (e.g. CLI/legacy imports).
    """
    portfolio, _ = Portfolio.objects.update_or_create(
        name=name,
        defaults={
            "total_capital": total_amount,
            "strategy": strategy_tag,
            "is_active": True,
            "owner": owner,
        }
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
                allocation_rs=h.get("allocation_rs", 0.0)
            )
        )
        
    if items_to_create:
        PortfolioItem.objects.bulk_create(items_to_create)
        
    logger.info(f"Successfully saved portfolio '{name}' with {len(items_to_create)} items to DB.")
    return portfolio

def load_portfolio_from_csv_to_db(csv_path: str, portfolio_name: str) -> Portfolio:
    """
    Utility to migrate legacy portfolio CSV files into the database.
    """
    df = pd.read_csv(csv_path)
    holdings = []
    total_val = 0.0
    
    for _, row in df.iterrows():
        val = row.get("quantity", 0) * row.get("purchase_price", 0)
        total_val += val
        holdings.append({
            "symbol": row.get("symbol"),
            "quantity": row.get("quantity", 0),
            "purchase_price": row.get("purchase_price", 0.0),
            "allocation_pct": 0.0, # Will be computed if missing
            "allocation_rs": val
        })
        
    # Normalize percentages
    for h in holdings:
        h["allocation_pct"] = (h["allocation_rs"] / total_val * 100) if total_val > 0 else 0.0
        
    return save_portfolio_to_db(portfolio_name, holdings, total_val, strategy_tag="legacy_csv")

