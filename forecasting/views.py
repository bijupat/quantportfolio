"""Views for the forecasting app: model training and composite reports."""

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.generic import TemplateView, View

from core.mixins import AdminRequiredMixin
from core.models import JobRun, Universe
from core.threading_utils import launch_tracked_command
from market_data.models import TrainedModel


class TrainModelView(AdminRequiredMixin, TemplateView):
    """Admin-only page for triggering train_model and viewing registered models/jobs."""

    template_name = "forecasting/train.html"

    def get_context_data(self, **kwargs) -> dict:
        """Adds seeded universes, previously registered models, and recent training jobs."""
        context = super().get_context_data(**kwargs)
        context["universes"] = Universe.objects.order_by("name")
        context["trained_models"] = TrainedModel.objects.order_by("-created_at")[:15]
        context["recent_jobs"] = JobRun.objects.filter(
            command_name__in=["train_model", "train_hybrid_model"]
        )[:10]
        return context


class TrainModelTriggerView(AdminRequiredMixin, View):
    """Launches `train_model` in the background with hyperparameters from the form."""

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Validates form input, starts the job, and renders the shared status partial."""
        model_name = request.POST.get("model_name", "").strip()
        universe = request.POST.get("universe")

        if not model_name or not universe:
            messages.error(request, "Model name and universe are required.")
            return redirect("forecasting:train")

        # Fail fast before spawning a thread — train_model.py uses get_or_create on
        # this name, so a silent collision would otherwise just skip retraining.
        if TrainedModel.objects.filter(name=model_name).exists():
            messages.error(request, f"A model named '{model_name}' already exists. Choose a different name.")
            return redirect("forecasting:train")

        try:
            options = {
                "model_name": model_name,
                "universe": universe,
                "start": request.POST.get("start") or "2015-01-01",
                "epochs": int(request.POST.get("epochs") or 50),
                "batch_size": int(request.POST.get("batch_size") or 64),
                "d_model": int(request.POST.get("d_model") or 64),
                "n_heads": int(request.POST.get("n_heads") or 4),
                "n_layers": int(request.POST.get("n_layers") or 2),
                "seq_len": int(request.POST.get("seq_len") or 60),
                "horizon": int(request.POST.get("horizon") or 30),
                "no_sentiment": request.POST.get("use_sentiment") != "on",
            }
        except ValueError:
            messages.error(request, "Epochs, batch size, and architecture fields must be whole numbers.")
            return redirect("forecasting:train")

        end = request.POST.get("end")
        if end:
            options["end"] = end

        job = launch_tracked_command("train_model", user=request.user, **options)
        return render(request, "core/_job_status.html", {"job": job})


class ReportsView(LoginRequiredMixin, TemplateView):
    """Composite/ensemble results and generated ReportArtifact downloads."""

    template_name = "forecasting/reports.html"