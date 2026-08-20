from django.contrib import admin
from market_data.models import (
    PriceBar,
    QualityScore,
    TrainedModel,
    TechnicalIndicatorSnapshot,
    NewsSentiment,
    MarketContext,
    FundamentalSnapshot,
    TrainingRun,
    NonTradingDay,
)

@admin.register(PriceBar)
class PriceBarAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'date', 'open', 'high', 'low', 'close', 'volume')
    list_filter = ('symbol',)
    date_hierarchy = 'date'

@admin.register(QualityScore)
class QualityScoreAdmin(admin.ModelAdmin):
    # Update these fields to match whatever attributes exist on your QualityScore model
    list_display = ('symbol', 'date', 'composite_score') 
    list_filter = ('symbol',)
    date_hierarchy = 'date'

@admin.register(TrainedModel)
class TrainedModelAdmin(admin.ModelAdmin):
    list_display = ('name', 'model_type', 'seq_len', 'horizon', 'd_model', 'created_at')
    list_filter = ('model_type', 'horizon')

@admin.register(TechnicalIndicatorSnapshot)
class TechnicalIndicatorSnapshotAdmin(admin.ModelAdmin):
    """Admin for per-symbol/date computed indicator snapshots (raw JSON payload in `values`)."""

    list_display = ('symbol', 'date')
    list_filter = ('symbol',)
    date_hierarchy = 'date'
    search_fields = ('symbol__ticker',)

@admin.register(NewsSentiment)
class NewsSentimentAdmin(admin.ModelAdmin):
    """Admin for DB-cached daily sentiment scores (see market_data.services.sentiment)."""

    list_display = (
        'symbol', 'date', 'sentiment_score', 'positive_count',
        'negative_count', 'news_volume', 'backend_used',
    )
    list_filter = ('backend_used', 'symbol')
    date_hierarchy = 'date'
    search_fields = ('symbol__ticker',)

@admin.register(MarketContext)
class MarketContextAdmin(admin.ModelAdmin):
    """Admin for per-index regime features (see market_data.services.market_context)."""

    list_display = ('index_ticker', 'date')
    list_filter = ('index_ticker',)
    date_hierarchy = 'date'

@admin.register(FundamentalSnapshot)
class FundamentalSnapshotAdmin(admin.ModelAdmin):
    """Admin for raw per-symbol fundamentals (placeholder — see market_data.services.fundamentals)."""

    list_display = ('symbol', 'fiscal_period', 'roe', 'roa', 'pe', 'revenue_growth', 'debt_equity', 'fcf')
    list_filter = ('fiscal_period',)
    search_fields = ('symbol__ticker',)

@admin.register(TrainingRun)
class TrainingRunAdmin(admin.ModelAdmin):
    """Admin for individual training-run attempts against a TrainedModel."""

    list_display = ('model', 'status', 'started_at', 'finished_at')
    list_filter = ('status', 'model')
    readonly_fields = ('started_at', 'finished_at')

@admin.register(NonTradingDay)
class NonTradingDayAdmin(admin.ModelAdmin):
    """Admin for the shared market-holiday cache (see market_data.services.prices).

    Read-mostly by design — see repair_price_history.py for the correct way to
    bulk-fix this table if it's ever found to contain bad entries; editing rows
    individually here should be a rare exception, not the normal workflow.
    """

    list_display = ('date', 'noted_at')
    date_hierarchy = 'date'
    readonly_fields = ('noted_at',)