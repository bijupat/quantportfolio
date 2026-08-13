"""URL routes for the market_data app."""

from django.urls import path

from market_data import views

app_name = "market_data"

urlpatterns = [
    path("fetch/", views.FetchDataView.as_view(), name="fetch"),
    path("fetch/seed/", views.SeedUniversesTriggerView.as_view(), name="fetch_seed"),
    path("fetch/prices/", views.FetchPricesTriggerView.as_view(), name="fetch_prices_trigger"),
]