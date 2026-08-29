from django.contrib.auth.models import AbstractUser
from django.db import models


class User(AbstractUser):
    """Custom user model extending Django's AbstractUser.

    Adds role-based access control and a default trading universe
    preference, used to gate management-command-backed views
    (e.g. train_model, run_composite) and personalize dashboards.
    """

    class Role(models.TextChoices):
        ANALYST = "analyst", "Analyst"
        ADMIN = "admin", "Admin"

    role: str = models.CharField(
        max_length=20,
        choices=Role.choices,
        default=Role.ANALYST,
        help_text="Controls access to training and portfolio-execution commands.",
    )
    default_universe = models.ForeignKey(
        "core.Universe",
        null=True,
        blank=True,
        on_delete=models.SET_NULL,
        related_name="default_for_users",
        help_text="Universe pre-selected in dashboard forms for this user.",
    )

    def __str__(self) -> str:
        return self.username

    @property
    def is_admin_role(self) -> bool:
        """Returns True if the user holds the admin role (distinct from is_superuser)."""
        return self.role == self.Role.ADMIN


class Symbol(models.Model):
    """A tradeable NSE/BSE equity.

    ``exchange`` is intentionally NOT a stored column — it's fully
    determined by the ticker's '.NS' / '.BO' suffix (the app's existing
    convention, e.g. 'RELIANCE.NS' vs 'RELIANCE.BO'). Storing it as a
    separate field would create a second source of truth that could
    drift from the ticker itself, so it's derived on access instead.
    """

    NSE = "NSE"
    BSE = "BSE"

    ticker: str = models.CharField(max_length=20, unique=True)
    name: str = models.CharField(max_length=100, blank=True, null=True)
    sector: str = models.CharField(max_length=50, blank=True, null=True)
    is_financial: bool = models.BooleanField(
        default=False,
        help_text="True for banks/NBFCs/insurers — used to exclude Altman "
                   "Z-Score and flag Piotroski interpretation caveats.",
    )
    is_active: bool = models.BooleanField(default=True)
    history_confirmed_start = models.DateField(
        null=True,
        blank=True,
        help_text="Earliest date for which yfinance has ever returned a real "
                   "PriceBar for this symbol, across all fetches so far. Set/"
                   "tightened by market_data.services.prices.get_price_bars — "
                   "once known, any future fetch requesting an earlier `start` "
                   "can skip straight to this date instead of re-querying "
                   "yfinance (and re-hitting its 'possibly delisted' warning) "
                   "for a pre-listing window already proven empty. Only ever "
                   "moved EARLIER by a new fetch that proves older data exists "
                   "than previously known — never blindly overwritten with a "
                   "later value, which would silently forget an earlier "
                   "confirmed start from a prior run. Null means no fetch has "
                   "confirmed anything yet (e.g. a brand-new Symbol row, or "
                   "one that's never been through get_price_bars). See "
                   "market_data's resync_history_start command to force "
                   "re-verification for a specific symbol if yfinance ever "
                   "surfaces older data than this field currently reflects "
                   "(e.g. a later corporate-action restatement).",
    )

    def __str__(self) -> str:
        return self.ticker

    @property
    def exchange(self) -> str:
        """Derived from the ticker suffix: '.BO' -> BSE, everything else -> NSE."""
        return self.BSE if self.ticker.endswith(".BO") else self.NSE


class Universe(models.Model):
    name = models.CharField(max_length=50, unique=True)
    description = models.TextField(blank=True, null=True)
    symbols = models.ManyToManyField(Symbol, through="UniverseMember", related_name="universes")

    def __str__(self) -> str:
        return self.name


class UniverseMember(models.Model):
    universe = models.ForeignKey(Universe, on_delete=models.CASCADE, related_name="members")
    symbol = models.ForeignKey(Symbol, on_delete=models.CASCADE)
    added_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        unique_together = ("universe", "symbol")

    def __str__(self) -> str:
        return f"{self.symbol.ticker} in {self.universe.name}"

class JobRun(models.Model):
    """Tracks a management command triggered asynchronously from the web UI.

    Lets an HTMX polling view report progress/output for long-running jobs
    (fetch_prices, seed_universes, train_model, run_composite) that would
    otherwise block the request thread that triggered them.

    ``progress_message``/``progress_current``/``progress_total`` exist to
    solve a specific gap: ``output`` is only populated once, in
    ``core.threading_utils.launch_tracked_command``'s ``finally`` block,
    after the entire management command has returned — a command that
    takes minutes (e.g. run_composite, ensembling several models across a
    large universe) shows nothing but a bare "Running" spinner for its
    whole duration, even though it internally passes through several
    well-defined stages. These three fields are updated *during* the run
    via ``core.threading_utils.update_job_progress()``, called explicitly
    by progress-aware commands (see ``PROGRESS_AWARE_COMMANDS``) at their
    existing stage boundaries, and are safe to poll from the HTMX status
    partial the same way ``status`` already is.

    Deliberately NOT reusing ``output`` for this: ``output`` is the
    captured stdout buffer, only ever written once at the end, and
    changing that write pattern to "flush periodically" would require
    threading a shared, lockable buffer through ``call_command`` — a much
    bigger change than three nullable fields updated via direct
    ``.update()`` calls (see ``update_job_progress``), which avoids ever
    saving a stale in-memory ``JobRun`` instance from within the
    background thread.
    """

    class Status(models.TextChoices):
        PENDING = "pending", "Pending"
        RUNNING = "running", "Running"
        SUCCESS = "success", "Success"
        FAILED = "failed", "Failed"

    command_name = models.CharField(max_length=100)
    options = models.JSONField(default=dict, blank=True)
    status = models.CharField(max_length=10, choices=Status.choices, default=Status.PENDING)
    output = models.TextField(blank=True)
    error = models.TextField(blank=True)
    progress_message = models.CharField(
        max_length=255,
        blank=True,
        help_text="Latest structured progress message (e.g. 'Step 2/5: Ensemble "
                   "scoring...'), updated during the run by progress-aware commands. "
                   "Blank for commands that don't report progress, or before the "
                   "first update.",
    )
    progress_current = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Current step number for progress-aware commands, e.g. 2 of 5. "
                   "Null if the command hasn't reported a step-based progress yet.",
    )
    progress_total = models.PositiveIntegerField(
        null=True,
        blank=True,
        help_text="Total step count for progress-aware commands, e.g. 5. Paired "
                   "with progress_current for a 'Step X/Y' display.",
    )
    triggered_by = models.ForeignKey(
        "core.User", null=True, blank=True, on_delete=models.SET_NULL, related_name="job_runs"
    )
    created_at = models.DateTimeField(auto_now_add=True)
    started_at = models.DateTimeField(null=True, blank=True)
    finished_at = models.DateTimeField(null=True, blank=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self) -> str:
        return f"{self.command_name} [{self.status}] #{self.pk}"

    @property
    def is_finished(self) -> bool:
        """Returns True once the job has reached a terminal (success/failed) state."""
        return self.status in (self.Status.SUCCESS, self.Status.FAILED)

    @property
    def has_progress(self) -> bool:
        """Returns True if this job has reported at least one progress update."""
        return bool(self.progress_message)