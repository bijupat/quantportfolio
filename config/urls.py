"""Root URL configuration for QuantPortfolioAI."""

from django.conf import settings
from django.conf.urls.static import static
from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path('admin/', admin.site.urls),
    path('', include('core.urls')),
    path('market/', include('market_data.urls')),
    path('forecasting/', include('forecasting.urls')),
    path('portfolio/', include('portfolio.urls')),
]

if settings.DEBUG:
    # Serve ReportArtifact PDFs/Excel files (media/reports/) in development.
    urlpatterns += static(settings.MEDIA_URL, document_root=settings.MEDIA_ROOT)