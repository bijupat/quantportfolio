from django.apps import AppConfig


class PortfolioConfig(AppConfig):
    """App configuration for the portfolio app (Portfolio, PortfolioItem models)."""

    default_auto_field = 'django.db.models.BigAutoField'
    name = 'portfolio'