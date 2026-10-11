from django.urls import path

from . import views

app_name = "inspection"

urlpatterns = [
    path("", views.home, name="home"),
    path("new/", views.new_inspection, name="new"),
    path("report/<int:pk>/", views.report, name="report"),
    path("report/<int:pk>/pdf/", views.report_pdf, name="report_pdf"),
    path("report/<int:pk>/delete/", views.delete_inspection, name="delete"),
]
