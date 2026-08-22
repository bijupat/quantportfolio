"""Views for the portfolio app: stored portfolio listing and holdings drill-down."""

from django.contrib.auth.mixins import LoginRequiredMixin
from django.db.models import Count, Q, QuerySet
from django.views.generic import DetailView, ListView

from portfolio.models import Portfolio, PortfolioStrategy


def _visible_portfolios_qs(user) -> QuerySet[Portfolio]:
    """Returns the Portfolios this user is allowed to see.

    Admins see every portfolio. Analysts see their own, ownerless/system-generated
    ones (e.g. CLI runs without --user-id, or legacy CSV imports), and any
    PortfolioStrategy.COMPOSITE_AI portfolio regardless of which admin triggered
    the run — composite-engine portfolios are shared analytical output, not
    personal data, even though Portfolio.owner still records who ran the job
    for audit purposes.
    """
    qs = Portfolio.objects.select_related("owner").annotate(holdings_count=Count("items"))
    qs = qs.order_by("-created_at")
    if user.is_admin_role:
        return qs
    return qs.filter(
        Q(owner=user) | Q(owner__isnull=True) | Q(strategy=PortfolioStrategy.COMPOSITE_AI)
    )


class PortfolioListView(LoginRequiredMixin, ListView):
    """Lists database-stored Portfolio records (manage_portfolio --list equivalent)."""

    model = Portfolio
    template_name = "portfolio/list.html"
    context_object_name = "portfolios"
    paginate_by = 20

    def get_queryset(self) -> QuerySet[Portfolio]:
        """Restricts the list to portfolios visible to the requesting user.

        The queryset is pre-annotated with holdings_count (see
        _visible_portfolios_qs) so the template can read {{ p.holdings_count }}
        instead of {{ p.item_count }} — avoiding one COUNT(*) query per row
        that Portfolio.item_count would otherwise issue across a full page
        of results.
        """
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