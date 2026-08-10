from django.contrib import admin
from forecasting.models import Prediction, ReportArtifact

@admin.register(Prediction)
class PredictionAdmin(admin.ModelAdmin):
    list_display = ('symbol', 'model', 'as_of_date', 'predicted_return', 'confidence')
    list_filter = ('model', 'as_of_date')
    date_hierarchy = 'as_of_date'

@admin.register(ReportArtifact)
class ReportArtifactAdmin(admin.ModelAdmin):
    list_display = ('kind', 'created_at', 'file')
    list_filter = ('kind', 'created_at')