from django.contrib import admin
from django.contrib.auth.admin import UserAdmin

from core.models import Symbol, Universe, UniverseMember, User
from core.forms import CustomUserCreationForm, CustomUserChangeForm


@admin.register(User)
class CustomUserAdmin(UserAdmin):
    """Admin configuration for the custom User model."""

    add_form = CustomUserCreationForm
    form = CustomUserChangeForm
    model = User

    list_display = ("username", "email", "role", "default_universe", "is_staff", "is_active")
    list_filter = ("role", "is_staff", "is_active", "default_universe")

    fieldsets = UserAdmin.fieldsets + (
        ("QuantPortfolioAI Settings", {"fields": ("role", "default_universe")}),
    )
    add_fieldsets = UserAdmin.add_fieldsets + (
        ("QuantPortfolioAI Settings", {"fields": ("role", "default_universe")}),
    )


@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ("ticker", "name", "sector", "is_active")
    search_fields = ("ticker", "name", "sector")
    list_filter = ("is_active", "sector")


class UniverseMemberInline(admin.TabularInline):
    model = UniverseMember
    extra = 1


@admin.register(Universe)
class UniverseAdmin(admin.ModelAdmin):
    list_display = ("name", "description")
    search_fields = ("name",)
    inlines = [UniverseMemberInline]