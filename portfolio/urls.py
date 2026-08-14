"""URL routes for the portfolio app."""

from django.urls import path

from portfolio import views

app_name = "portfolio"

urlpatterns = [
    path("", views.PortfolioListView.as_view(), name="list"),
    path("<int:pk>/", views.PortfolioDetailView.as_view(), name="detail"),
]