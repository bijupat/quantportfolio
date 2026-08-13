"""Reusable view access-control mixins shared across apps."""

from django.contrib.auth.mixins import LoginRequiredMixin, UserPassesTestMixin


class AdminRequiredMixin(LoginRequiredMixin, UserPassesTestMixin):
    """Restricts a class-based view to authenticated users with the Admin role.

    Analysts hitting an admin-only view get a 403 rather than a redirect loop,
    since they *are* authenticated — they're just not permitted.
    """

    def test_func(self) -> bool:
        """Return True only for authenticated users flagged as admin_role."""
        return self.request.user.is_authenticated and self.request.user.is_admin_role