import threading
import logging
from django.core.management import call_command

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