import threading
import logging
from django.core.management import call_command
import io
from django.utils import timezone


logger = logging.getLogger(__name__)

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



def launch_tracked_command(command_name: str, user=None, **options) -> JobRun:
    """Creates a JobRun row and executes a management command in a background thread.

    Captures stdout/stderr into the JobRun so an HTMX polling view can render
    live progress and final output without blocking the triggering request.

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
        try:
            call_command(command_name, stdout=out, stderr=err, **options)
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