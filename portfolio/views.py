"""Views for the portfolio app: stored portfolio listing and holdings drill-down."""

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Q, QuerySet
from django.views.generic import DetailView, ListView

from portfolio.models import Portfolio


def _visible_portfolios_qs(user) -> QuerySet[Portfolio]:
    """Returns the Portfolios this user is allowed to see.

    Admins see every portfolio. Analysts see only their own plus
    ownerless/system-generated portfolios (owner=None) — e.g. those created
    via CLI/management commands before ownership wiring reaches those paths.
    """
    qs = Portfolio.objects.select_related("owner").order_by("-created_at")
    if user.is_admin_role:
        return qs
    return qs.filter(Q(owner=user) | Q(owner__isnull=True))

def _visible_portfolios_qs(user) -> QuerySet[Portfolio]:
    """Returns the Portfolios this user is allowed to see.

    Admins see every portfolio. Analysts see their own, ownerless/system-generated
    ones, and composite-engine-generated ones regardless of which admin triggered
    the run — composite portfolios are shared analytical output, not personal data,
    even though Portfolio.owner records who ran the job for audit purposes.
    """
    qs = Portfolio.objects.select_related("owner").order_by("-created_at")
    if user.is_admin_role:
        return qs
    return qs.filter(Q(owner=user) | Q(owner__isnull=True) | Q(strategy="composite_ai"))

class PortfolioListView(LoginRequiredMixin, ListView):
    """Lists database-stored Portfolio records (manage_portfolio --list equivalent)."""

    model = Portfolio
    template_name = "portfolio/list.html"
    context_object_name = "portfolios"
    paginate_by = 20

    def get_queryset(self) -> QuerySet[Portfolio]:
        """Restricts the list to portfolios visible to the requesting user."""
        return _visible_portfolios_qs(self.request.user)


class PortfolioDetailView(LoginRequiredMixin, DetailView):
    """Shows a single Portfolio's holdings (manage_portfolio --view equivalent)."""

    model = Portfolio
    template_name = "portfolio/detail.html"
    context_object_name = "portfolio"
    slug_field = "name"
    slug_url_kwarg = "name"

    def get_queryset(self) -> QuerySet[Portfolio]:
        """Restricts lookup to portfolios visible to the requesting user.

        Using this (rather than Portfolio.objects.all()) means a portfolio
        outside the user's visibility 404s instead of resolving — it never
        confirms the portfolio's existence to someone who shouldn't see it.
        """
        return _visible_portfolios_qs(self.request.user)

    def get_context_data(self, **kwargs) -> dict:
        """Adds the portfolio's holdings, ordered by allocation weight."""
        context = super().get_context_data(**kwargs)
        context["items"] = self.object.items.select_related("symbol").order_by("-allocation_pct")
        return context