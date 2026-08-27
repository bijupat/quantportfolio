"""Admin configuration for the portfolio app: Portfolio and PortfolioItem.

Registers both models with list/search/filter configuration tuned for the
project's actual data scale — see PortfolioItemInline and PortfolioItemAdmin
below for why Symbol/Portfolio foreign keys use autocomplete_fields rather
than the default <select> widget: seed_universes.py seeds 500+ Symbol rows
(nifty500 alone), and a default FK dropdown renders every option server-side
on every page load.
"""

from django.contrib import admin
from django.db.models import Count, QuerySet
from django.http import HttpRequest

from portfolio.models import Portfolio, PortfolioItem


class PortfolioItemInline(admin.TabularInline):
    """Inline editor for holdings within a Portfolio in the admin.

    Uses autocomplete_fields for `symbol` rather than the default <select> —
    with 500+ seeded Symbol rows (see seed_universes.py's nifty500 universe),
    a plain dropdown renders every ticker on every page load. Requires
    SymbolAdmin (core/admin.py) to declare search_fields, which it already
    does (`ticker`, `name`, `sector`).

    Numeric fields (quantity, purchase_price, allocation_pct, allocation_rs)
    are guarded by both MinValueValidator/MaxValueValidator (see
    portfolio/models.py) and database CheckConstraints. The validators run
    via this inline's ModelForm.full_clean() on save, so an admin typing a
    negative quantity here gets a normal in-form validation error rather
    than an unhandled IntegrityError — unlike the bulk_create() paths in
    portfolio.services.portfolio_service, which bypass validators entirely
    and rely on the CheckConstraints alone.
    """

    model = PortfolioItem
    extra = 0
    fields = ("symbol", "quantity", "purchase_price", "allocation_pct", "allocation_rs")
    autocomplete_fields = ("symbol",)


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    """Admin configuration for Portfolio, including owner and holdings inline."""

    list_display = (
        "name", "owner", "strategy", "total_capital",
        "holdings_count", "is_active", "created_at",
    )
    list_filter = ("strategy", "is_active", "owner")
    search_fields = ("name", "owner__username")
    date_hierarchy = "created_at"
    readonly_fields = ("created_at",)
    autocomplete_fields = ("owner",)
    inlines = [PortfolioItemInline]

    def get_queryset(self, request: HttpRequest) -> QuerySet[Portfolio]:
        """Annotates each row with its holdings count in a single query.

        Mirrors portfolio.views._visible_portfolios_qs's Count("items")
        annotation — without this, holdings_count() below would issue one
        extra COUNT(*) query per row (N+1), the same issue already fixed in
        the front-end portfolio list view.
        """
        qs = super().get_queryset(request)
        return qs.annotate(_holdings_count=Count("items"))

    @admin.display(description="Holdings", ordering="_holdings_count")
    def holdings_count(self, obj: Portfolio) -> int:
        """Returns the annotated holdings count for list_display.

        Reads the `_holdings_count` annotation added by get_queryset() above
        rather than calling obj.items.count() or the Portfolio.item_count
        property, either of which would issue a fresh query per row.
        """
        return obj._holdings_count


@admin.register(PortfolioItem)
class PortfolioItemAdmin(admin.ModelAdmin):
    """Admin configuration for individual PortfolioItem rows."""

    list_display = (
        "portfolio", "symbol", "quantity", "purchase_price",
        "allocation_pct", "allocation_rs",
    )
    list_filter = ("portfolio",)
    search_fields = ("symbol__ticker", "portfolio__name")
    autocomplete_fields = ("portfolio", "symbol")