"""Views for the portfolio app: stored portfolio listing."""

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView


class PortfolioListView(LoginRequiredMixin, TemplateView):
    """Lists database-stored Portfolio records (manage_portfolio --list equivalent)."""

    template_name = "portfolio/list.html"