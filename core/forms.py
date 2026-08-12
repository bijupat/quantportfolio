from django import forms
from django.contrib.auth.forms import UserCreationForm, UserChangeForm

from core.models import User


class CustomUserCreationForm(UserCreationForm):
    """Form for creating new users in Django admin, including role and default_universe."""

    class Meta(UserCreationForm.Meta):
        model = User
        fields = ("username", "email", "role", "default_universe")


class CustomUserChangeForm(UserChangeForm):
    """Form for editing existing users in Django admin, including role and default_universe."""

    class Meta(UserChangeForm.Meta):
        model = User
        fields = (
            "username", "email", "first_name", "last_name",
            "role", "default_universe", "is_active", "is_staff", "is_superuser",
            "groups", "user_permissions",
        )