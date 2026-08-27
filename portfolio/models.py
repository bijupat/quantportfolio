from django.conf import settings
from django.core.validators import MaxValueValidator, MinValueValidator
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
    total_capital = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Total rupee capital represented by this portfolio's holdings. "
                   "Cannot be negative.",
    )
    strategy = models.CharField(
        max_length=50,
        choices=PortfolioStrategy.choices,
        default=PortfolioStrategy.MANUAL,
    )
    is_active = models.BooleanField(default=True)
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]
        constraints = [
            models.CheckConstraint(
                # Named check=... (not condition=...) for Django <5.1 compatibility;
                # both kwargs are accepted as of 5.0, but `condition` only became the
                # non-deprecated name in 5.1 — this project pins Django==5.0.14.
                check=models.Q(total_capital__gte=0),
                name="portfolio_total_capital_non_negative",
            ),
        ]

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
    """A single holding (symbol + quantity + allocation) within a Portfolio.

    Numeric fields are guarded two ways:
      - MinValueValidator/MaxValueValidator, enforced by full_clean() — covers
        ModelForm and Django admin input.
      - Meta.constraints (CheckConstraint), enforced by the database itself on
        every INSERT/UPDATE — covers bulk_create() and bulk_update(), which
        skip validators entirely. Both of this app's current writers
        (portfolio.services.portfolio_service.save_portfolio_to_db and,
        transitively, forecasting.services.composite.build_composite_portfolio)
        write via bulk_create(), so the DB constraint is what actually protects
        this data today; the validators are for the admin/future-form path.

    quantity=0 and purchase_price=0.0 are valid, not just theoretically
    allowed: build_composite_portfolio explicitly produces qty=0 and
    price=0.0 (price_source="unavailable") when a symbol has no cached
    PriceBar in the lookup window — see forecasting/services/composite.py.
    Validators/constraints therefore use gte=0, not gte=1, to avoid rejecting
    that legitimate, already-shipping fallback state.
    """

    portfolio = models.ForeignKey(Portfolio, on_delete=models.CASCADE, related_name="items")
    symbol = models.ForeignKey("core.Symbol", on_delete=models.CASCADE)
    quantity = models.IntegerField(
        default=0,
        validators=[MinValueValidator(0)],
        help_text="Number of shares held. 0 is valid (e.g. price lookup failed at "
                   "allocation time); negative values are not.",
    )
    purchase_price = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Price per share at purchase, in rupees. 0.0 is valid and signals "
                   "an unavailable price (see build_composite_portfolio); negative "
                   "values are not.",
    )
    allocation_pct = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0), MaxValueValidator(100.0)],
        help_text="Share of total portfolio capital allocated to this holding, as a "
                   "percentage (0-100).",
    )
    allocation_rs = models.FloatField(
        default=0.0,
        validators=[MinValueValidator(0.0)],
        help_text="Rupee amount allocated to this holding. Cannot be negative.",
    )

    class Meta:
        unique_together = ("portfolio", "symbol")
        constraints = [
            models.CheckConstraint(
                check=models.Q(quantity__gte=0),
                name="portfolioitem_quantity_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(purchase_price__gte=0),
                name="portfolioitem_purchase_price_non_negative",
            ),
            models.CheckConstraint(
                check=models.Q(allocation_pct__gte=0) & models.Q(allocation_pct__lte=100),
                name="portfolioitem_allocation_pct_in_range",
            ),
            models.CheckConstraint(
                check=models.Q(allocation_rs__gte=0),
                name="portfolioitem_allocation_rs_non_negative",
            ),
        ]

    def __str__(self) -> str:
        return f"{self.symbol.ticker} ({self.quantity}) in {self.portfolio.name}"

    @property
    def market_value(self) -> float:
        """Returns quantity * purchase_price for this holding."""
        return self.quantity * self.purchase_price