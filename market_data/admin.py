from django.contrib import admin
from market_data.models import PriceBar, QualityScore, TrainedModel

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
    list_display = ('name', 'model_type', 'seq_len', 'd_model', 'created_at')
    list_filter = ('model_type',)