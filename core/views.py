"""Views for the core app: dashboard landing page and auth."""

from django.contrib.auth.mixins import LoginRequiredMixin
from django.views.generic import TemplateView
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse
from django.shortcuts import get_object_or_404, render
from django.views import View
from core.models import JobRun


class DashboardView(LoginRequiredMixin, TemplateView):
    """Landing page / overview shown after login."""

    template_name = "core/index.html"




class JobStatusView(LoginRequiredMixin, View):
    """Renders the current status fragment for a JobRun — polled by HTMX until finished.

    Shared across apps (market_data, forecasting, ...) so each trigger view only
    needs to call launch_tracked_command() and point its form at this one endpoint,
    instead of every app reimplementing its own status partial.
    """

    def get(self, request: HttpRequest, job_id: int, *args, **kwargs) -> HttpResponse:
        """Fetches the job and re-renders the shared partial with its latest state."""
        job = get_object_or_404(JobRun, pk=job_id)
        return render(request, "core/_job_status.html", {"job": job})