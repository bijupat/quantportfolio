"""URL routes for the core app: dashboard + authentication."""

from django.contrib.auth import views as auth_views
from django.urls import path

from core import views

app_name = "core"

urlpatterns = [
    path("", views.DashboardView.as_view(), name="index"),
    path(
        "login/",
        auth_views.LoginView.as_view(template_name="registration/login.html"),
        name="login",
    ),
    path("logout/", auth_views.LogoutView.as_view(), name="logout"),
    path("jobs/<int:job_id>/status/", views.JobStatusView.as_view(), name="job_status"),
]