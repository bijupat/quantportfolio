"""Views for the forecasting app: model training, composite reports, and the screener."""

from django.contrib import messages
from django.contrib.auth.mixins import LoginRequiredMixin
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.generic import TemplateView, View

from core.mixins import AdminRequiredMixin
from core.models import JobRun, Universe
from core.threading_utils import launch_tracked_command
from forecasting.models import ReportArtifact
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


class TrainHybridModelTriggerView(AdminRequiredMixin, View):
    """Launches `train_hybrid_model` in the background with hyperparameters from the form."""

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Validates form input, starts the job, and renders the shared status partial."""
        model_name = request.POST.get("model_name", "").strip()
        universe = request.POST.get("universe")

        if not model_name or not universe:
            messages.error(request, "Model name and universe are required.")
            return redirect("forecasting:train")

        if TrainedModel.objects.filter(name=model_name).exists():
            messages.error(request, f"A model named '{model_name}' already exists. Choose a different name.")
            return redirect("forecasting:train")

        try:
            options = {
                "model_name": model_name,
                "universe": universe,
                "start": request.POST.get("start") or "2015-01-01",
                "epochs": int(request.POST.get("epochs") or 60),
                "batch_size": int(request.POST.get("batch_size") or 32),
                "d_model": int(request.POST.get("d_model") or 64),
                "n_heads": int(request.POST.get("n_heads") or 4),
                "n_layers": int(request.POST.get("n_layers") or 2),
                "seq_len": int(request.POST.get("seq_len") or 120),
                "lstm_units": int(request.POST.get("lstm_units") or 64),
                "no_sentiment": request.POST.get("use_sentiment") != "on",
            }
        except ValueError:
            messages.error(request, "Epochs, batch size, and architecture fields must be whole numbers.")
            return redirect("forecasting:train")

        end = request.POST.get("end")
        if end:
            options["end"] = end

        job = launch_tracked_command("train_hybrid_model", user=request.user, **options)
        return render(request, "core/_job_status.html", {"job": job})


class ReportsView(LoginRequiredMixin, TemplateView):
    """Composite/ensemble results, the screener, and generated ReportArtifact downloads."""

    template_name = "forecasting/reports.html"

    def get_context_data(self, **kwargs) -> dict:
        """Adds registered models, universes, recent composite/screener jobs, and generated reports."""
        context = super().get_context_data(**kwargs)
        context["trained_models"] = TrainedModel.objects.order_by("-created_at")
        context["universes"] = Universe.objects.order_by("name")
        context["recent_jobs"] = JobRun.objects.filter(command_name="run_composite")[:10]
        context["recent_screener_jobs"] = JobRun.objects.filter(command_name="run_screener")[:10]
        context["report_artifacts"] = ReportArtifact.objects.select_related(
            "portfolio"
        ).order_by("-created_at")[:20]
        return context


class RunCompositeTriggerView(AdminRequiredMixin, View):
    """Launches `run_composite` in the background with ensemble/weighting settings from the form."""

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Validates form input, pairs each selected model with its weight, and starts the job."""
        model_names = request.POST.getlist("models")
        universe = request.POST.get("universe")

        if not model_names:
            messages.error(request, "Select at least one trained model.")
            return redirect("forecasting:reports")
        if not universe:
            messages.error(request, "Universe is required.")
            return redirect("forecasting:reports")

        # Pair each selected model with its own weight input (name="weight__<model_name>")
        # rather than a separate comma-list — keeps the two lists correctly aligned
        # regardless of checkbox order.
        try:
            model_weights = [
                float(request.POST.get(f"weight__{name}", "1.0")) for name in model_names
            ]
        except ValueError:
            messages.error(request, "Model weights must be valid numbers.")
            return redirect("forecasting:reports")

        try:
            options = {
                "models": model_names,
                "model_weights": model_weights,
                "universe": universe,
                "top_n": int(request.POST.get("top_n") or 10),
                "amount": float(request.POST.get("amount") or 100000.0),
                "w_transformer": float(request.POST.get("w_transformer") or 0.80),
                "w_quality": float(request.POST.get("w_quality") or 0.12),
                "w_technical": float(request.POST.get("w_technical") or 0.08),
                "no_sentiment": request.POST.get("use_sentiment") != "on",
                "plot": request.POST.get("save_reports") == "on",
                "user_id": request.user.id,
            }
        except ValueError:
            messages.error(request, "Top-N, amount, and layer-weight fields must be valid numbers.")
            return redirect("forecasting:reports")

        as_of = request.POST.get("as_of")
        if as_of:
            options["as_of"] = as_of

        job = launch_tracked_command("run_composite", user=request.user, **options)
        return render(request, "core/_job_status.html", {"job": job})


class RunScreenerTriggerView(LoginRequiredMixin, View):
    """Launches `run_screener` in the background with filter settings from the form.

    Unlike RunCompositeTriggerView (AdminRequiredMixin) and the training
    triggers, the screener is a read-only, side-effect-free filtering pass
    over existing DB-cached price data (see forecasting/services/screener.py)
    — no model training, no portfolio persistence — so it's gated the same
    as the Composite Reports page itself (any logged-in user), not
    admin-only.
    """

    def post(self, request: HttpRequest, *args, **kwargs) -> HttpResponse:
        """Validates form input, starts the job, and renders the shared status partial."""
        universe = request.POST.get("screener_universe")
        symbols_raw = request.POST.get("screener_symbols", "").strip()

        if not universe and not symbols_raw:
            messages.error(request, "Provide a universe or a specific symbol list for the screener.")
            return redirect("forecasting:reports")

        try:
            options = {
                "top_n": int(request.POST.get("screener_top_n") or 30),
                "min_score": int(request.POST.get("screener_min_score") or 3),
                "verbose": request.POST.get("screener_verbose") == "on",
                "save_csv": request.POST.get("screener_save_csv") == "on",
            }
        except ValueError:
            messages.error(request, "Top-N and min-score must be whole numbers.")
            return redirect("forecasting:reports")

        # --symbols overrides --universe in the command itself (see
        # run_screener.py's handle()), so only one needs to be passed —
        # prefer an explicit symbol list when both are somehow present.
        if symbols_raw:
            options["symbols"] = [s.strip() for s in symbols_raw.split(",") if s.strip()]
        else:
            options["universe"] = universe

        as_of = request.POST.get("screener_as_of")
        if as_of:
            options["as_of"] = as_of

        job = launch_tracked_command("run_screener", user=request.user, **options)
        return render(request, "core/_job_status.html", {"job": job})