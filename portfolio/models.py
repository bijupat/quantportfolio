from django.conf import settings
from django.db import models

from core.models import Symbol


class Portfolio(models.Model):
    """Represents a saved, named portfolio owned by a specific user.

    Portfolios are produced either by the composite/ensemble pipeline
    (forecasting.services.composite.build_composite_portfolio) or
    imported from legacy CSVs via portfolio_service.load_portfolio_from_csv_to_db.
    """

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.CASCADE,
        related_name="portfolios",
        null=True,
        blank=True,
        help_text="User who owns this portfolio. Null for legacy/system-generated portfolios.",
    )
    name = models.CharField(max_length=100, unique=True)
    total_capital = models.FloatField(default=0.0)
    strategy = models.CharField(max_length=50, default="composite")
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def item_count(self) -> int:
        """Returns the number of holdings in this portfolio."""
        return self.items.count()


class PortfolioItem(models.Model):
    """A single holding (symbol + quantity + allocation) within a Portfolio."""

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="items")
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    quantity = models.IntegerField(default=0)
    purchase_price = models.FloatField(default=0.0)
    allocation_pct = models.FloatField(default=0.0)
    allocation_rs = models.FloatField(default=0.0)

    class Meta:
        unique_together = ("portfolio", "symbol")

    def __str__(self) -> str:
        return f"{self.symbol.ticker} ({self.quantity}) in {self.portfolio.name}"

    @property
    def market_value(self) -> float:
        """Returns quantity * purchase_price for this holding."""
        return self.quantity * self.purchase_price