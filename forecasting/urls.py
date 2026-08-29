"""URL routes for the forecasting app."""

from django.urls import path

from forecasting import views

app_name = "forecasting"

urlpatterns = [
    path("train/", views.TrainModelView.as_view(), name="train"),
    path("train/run/", views.TrainModelTriggerView.as_view(), name="train_trigger"),
    path("train/run-hybrid/", views.TrainHybridModelTriggerView.as_view(), name="train_hybrid_trigger"),
    path("reports/", views.ReportsView.as_view(), name="reports"),
    path("reports/run/", views.RunCompositeTriggerView.as_view(), name="reports_run"),
    path("reports/screener/run/", views.RunScreenerTriggerView.as_view(), name="screener_run"),
]