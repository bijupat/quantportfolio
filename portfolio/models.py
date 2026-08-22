from django.conf import settings
from django.db import models


class PortfolioStrategy(models.TextChoices):
    """Canonical strategy tags written by every portfolio-producing code path.

    Centralising these avoids the same string being duplicated (and
    potentially typo'd or renamed out of sync) across the writer
    (forecasting.management.commands.run_composite,
    portfolio.services.portfolio_service) and the reader
    (portfolio.views._visible_portfolios_qs, which filters on
    COMPOSITE_AI to grant analysts read access to composite-engine output
    regardless of which admin triggered the run).
    """

    COMPOSITE_AI = "composite_ai", "Composite AI Engine"
    LEGACY_CSV = "legacy_csv", "Legacy CSV Import"
    MANUAL = "manual", "Manual"


class Portfolio(models.Model):
    """Represents a saved, named portfolio owned by a specific user.

    Portfolios are produced either by the composite/ensemble pipeline
    (forecasting.services.composite.build_composite_portfolio) or
    imported from legacy CSVs via portfolio_service.load_portfolio_from_csv_to_db.
    """

    owner = models.ForeignKey(
        settings.AUTH_USER_MODEL,
        on_delete=models.SET_NULL,
        related_name="portfolios",
        null=True,
        blank=True,
        help_text="User who owns this portfolio. Null for legacy/system-generated "
                   "portfolios, or after the triggering user's account is deleted. "
                   "Deliberately SET_NULL (not CASCADE): composite-engine portfolios "
                   "are shared analytical output visible to every analyst (see "
                   "portfolio.views._visible_portfolios_qs), not personal data tied "
                   "to the admin who happened to trigger the run — deleting that "
                   "admin's account must not delete portfolios other users rely on.",
    )
    name = models.CharField(max_length=100, unique=True)
    total_capital = models.FloatField(default=0.0)
    strategy = models.CharField(
        max_length=50,
        choices=PortfolioStrategy.choices,
        default=PortfolioStrategy.MANUAL,
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return self.name

    @property
    def item_count(self) -> int:
        """Returns the number of holdings in this portfolio.

        Note: prefer annotating with Count("items") in list views instead of
        relying on this property in a loop — each access here issues its own
        COUNT(*) query (see portfolio.views.PortfolioListView).
        """
        return self.items.count()


class PortfolioItem(models.Model):
    """A single holding (symbol + quantity + allocation) within a Portfolio."""

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="items")
    symbol = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
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