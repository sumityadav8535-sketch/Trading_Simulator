from django.contrib import admin
from django.urls import include, path

urlpatterns = [
    path("admin/", admin.site.urls),
    path("stage-analysis/", include("stage_analysis.urls")),
    path("stage-analysis-v2/", include("stage_analysis_v2.urls")),
    path("", include("trading.urls")),
]