from django.http import HttpRequest
from django.core.exceptions import PermissionDenied

from core.models import User


def require_role(request: HttpRequest, *allowed_roles: str) -> None:
    """Raises PermissionDenied unless request.user.role is one of allowed_roles.

    Usage:
        require_role(request, User.Role.ADMIN)
        require_role(request, User.Role.ADMIN, User.Role.ANALYST)
    """
    if not request.user.is_authenticated:
        raise PermissionDenied("Authentication required.")
    if request.user.role not in allowed_roles:
        raise PermissionDenied(
            f"Role '{request.user.role}' is not permitted to perform this action."
        )