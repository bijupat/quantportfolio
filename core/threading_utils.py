import threading
import logging
from django.core.management import call_command
import io
from django.utils import timezone


logger = logging.getLogger(__name__)

# Commands that accept a --job-id argument and call update_job_progress()
# internally to report structured progress mid-run (see JobRun's docstring
# in core/models.py for why this exists). An explicit allowlist — rather
# than always passing job_id=job.pk into every call_command() — because
# Django's call_command raises TypeError for any kwarg a command's parser
# doesn't declare; blindly passing job_id to every tracked command would
# break every command that hasn't opted in. Add a command's name here only
# after it defines --job-id in add_arguments() and calls
# update_job_progress() at its own stage boundaries.
PROGRESS_AWARE_COMMANDS = {"run_composite"}


def run_in_background(target_func, *args, **kwargs):
    """
    Spawns a lightweight daemon thread to run any function asynchronously.
    """
    def wrapper():
        try:
            target_func(*args, **kwargs)
        except Exception as e:
            logger.error(f"Error in background thread {target_func.__name__}: {e}", exc_info=True)

    thread = threading.Thread(target=wrapper, daemon=True)
    thread.start()
    logger.info(f"Background thread started for: {target_func.__name__}")
    return thread

def run_management_command_in_background(command_name, **options):
    """
    Spawns a background thread to execute a Django management command safely.
    """
    def task():
        try:
            logger.info(f"Starting background command: {command_name} with {options}")
            call_command(command_name, **options)
            logger.info(f"Successfully finished background command: {command_name}")
        except Exception as e:
            logger.error(f"Error running command {command_name} in background: {e}", exc_info=True)

    thread = threading.Thread(target=task, daemon=True)
    thread.start()
    return thread


# NOTE: import placed here (not top-level) to avoid a circular import,
# since core.models has no dependency on this module.
from core.models import JobRun


def update_job_progress(job_id: int, message: str, current: int | None = None, total: int | None = None) -> None:
    """Updates a JobRun's progress fields immediately, visible to the next HTMX poll.

    Called by progress-aware management commands (see PROGRESS_AWARE_COMMANDS)
    from inside the background thread launch_tracked_command() spawns, at
    whatever stage boundaries the command already has. Intentionally uses a
    direct queryset .update() rather than fetching the JobRun instance and
    calling .save() — the latter would risk overwriting status/output/
    finished_at fields concurrently being written by launch_tracked_command's
    own task() closure with a stale in-memory snapshot, since both would be
    running in the same thread sequentially but a future change could easily
    introduce a race if this pattern were copied elsewhere. .update() only
    ever touches the three columns named here, and is silently a no-op if
    the JobRun no longer exists (e.g. deleted mid-run) rather than raising.

    Args:
        job_id: Primary key of the JobRun this progress update belongs to.
        message: Human-readable status, e.g. "Step 2/5: Ensemble scoring...".
        current: Optional current step number, for a "Step X/Y" display.
        total: Optional total step count, paired with current.
    """
    JobRun.objects.filter(pk=job_id).update(
        progress_message=message,
        progress_current=current,
        progress_total=total,
    )


def launch_tracked_command(command_name: str, user=None, **options) -> JobRun:
    """Creates a JobRun row and executes a management command in a background thread.

    Captures stdout/stderr into the JobRun so an HTMX polling view can render
    live progress and final output without blocking the triggering request.

    For commands listed in PROGRESS_AWARE_COMMANDS, the JobRun's own primary
    key is also passed through as --job-id, letting the command report
    structured progress mid-run via update_job_progress() — see that
    function and JobRun's docstring (core/models.py) for the full rationale.
    Commands not in the allowlist are invoked exactly as before, unaffected.

    Args:
        command_name: Name of the registered management command (e.g. 'fetch_prices').
        user: The requesting core.User, recorded as triggered_by (or None/anonymous).
        **options: Keyword arguments forwarded to call_command, matching the
            command's argparse dest names (e.g. universe='nifty50', start='2020-01-01').

    Returns:
        The JobRun row immediately (status=PENDING) — the caller should render
        it and let the client poll core.JobRun's status endpoint for updates.
    """
    job = JobRun.objects.create(
        command_name=command_name,
        options=options,
        status=JobRun.Status.PENDING,
        triggered_by=user if getattr(user, "is_authenticated", False) else None,
    )

    def task() -> None:
        job.status = JobRun.Status.RUNNING
        job.started_at = timezone.now()
        job.save(update_fields=["status", "started_at"])

        out, err = io.StringIO(), io.StringIO()
        call_kwargs = dict(options)
        if command_name in PROGRESS_AWARE_COMMANDS:
            call_kwargs["job_id"] = job.pk

        try:
            call_command(command_name, stdout=out, stderr=err, **call_kwargs)
            job.status = JobRun.Status.SUCCESS
        except Exception as e:
            err.write(f"\n{e}")
            job.status = JobRun.Status.FAILED
            logger.error(f"Tracked command '{command_name}' (job #{job.pk}) failed: {e}", exc_info=True)
        finally:
            job.output = out.getvalue()
            job.error = err.getvalue()
            job.finished_at = timezone.now()
            job.save(update_fields=["status", "output", "error", "finished_at"])

    thread = threading.Thread(target=task, daemon=True)
    thread.start()
    return job