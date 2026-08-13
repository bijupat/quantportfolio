"""Views for the market_data app: data pipeline trigger UI and job polling."""

from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.generic import TemplateView, View

from core.mixins import AdminRequiredMixin
from core.models import JobRun, Universe
from core.threading_utils import launch_tracked_command


class FetchDataView(AdminRequiredMixin, TemplateView):
    """Admin-only page for triggering seed_universes / fetch_prices and viewing recent jobs."""

    template_name = "market_data/fetch.html"

    def get_context_data(self, **kwargs) -> dict:
        """Adds seeded universes (for the select input) and recent job history."""
        context = super().get_context_data(**kwargs)
        context["universes"] = Universe.objects.order_by("name")
        context["recent_jobs"] = JobRun.objects.filter(
            command_name__in=["seed_universes", "fetch_prices"]
        )[:10]
        return context


class SeedUniversesTriggerView(AdminRequiredMixin, View):
    """Launches `seed_universes` in the background; returns a pollable status fragment."""

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Starts the job and renders the initial (pending) status partial."""
        job = launch_tracked_command("seed_universes", user=request.user)
        return render(request, "core/_job_status.html", {"job": job})


class FetchPricesTriggerView(AdminRequiredMixin, View):
    """Launches `fetch_prices --universe --start` in the background."""

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Validates form input, starts the job, and renders the status partial."""
        universe_name = request.POST.get("universe")
        start = request.POST.get("start")

        if not universe_name or not start:
            messages.error(request, "Universe and start date are required.")
            return redirect("market_data:fetch")

        job = launch_tracked_command(
            "fetch_prices", user=request.user, universe=universe_name, start=start
        )
        return render(request, "core/_job_status.html", {"job": job})