from django.contrib import admin

from portfolio.models import Portfolio, PortfolioItem


class PortfolioItemInline(admin.TabularInline):
    """Inline editor for holdings within a Portfolio in the admin."""

    model = PortfolioItem
    extra = 0
    fields = ("symbol", "quantity", "purchase_price", "allocation_pct", "allocation_rs")
    readonly_fields = ()


@admin.register(Portfolio)
class PortfolioAdmin(admin.ModelAdmin):
    """Admin configuration for Portfolio, including owner and holdings inline."""

    list_display = ("name", "owner", "strategy", "total_capital", "is_active", "created_at")
    list_filter = ("strategy", "is_active", "owner")
    search_fields = ("name", "owner__username")
    date_hierarchy = "created_at"
    inlines = [PortfolioItemInline]


@admin.register(PortfolioItem)
class PortfolioItemAdmin(admin.ModelAdmin):
    """Admin configuration for individual PortfolioItem rows."""

    list_display = ("portfolio", "symbol", "quantity", "purchase_price", "allocation_pct")
    list_filter = ("portfolio",)
    search_fields = ("symbol__ticker", "portfolio__name")