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
    ticker = models.CharField(max_length=20, unique=True)
    name = models.CharField(max_length=100, blank=True, null=True)
    sector = models.CharField(max_length=50, blank=True, null=True)
    is_active = models.BooleanField(default=True)

    def __str__(self) -> str:
        return self.ticker


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