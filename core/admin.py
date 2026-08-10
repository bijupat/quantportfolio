from django.contrib import admin
from core.models import Symbol, Universe, UniverseMember

@admin.register(Symbol)
class SymbolAdmin(admin.ModelAdmin):
    list_display = ('ticker', 'name', 'sector', 'is_active')
    search_fields = ('ticker', 'name', 'sector')
    list_filter = ('is_active', 'sector')

class UniverseMemberInline(admin.TabularInline):
    model = UniverseMember
    extra = 1

@admin.register(Universe)
class UniverseAdmin(admin.ModelAdmin):
    list_display = ('name', 'description')
    search_fields = ('name',)
    inlines = [UniverseMemberInline]