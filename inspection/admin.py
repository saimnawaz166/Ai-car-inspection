from django.contrib import admin

from .models import Detection, Inspection, ZoneCheck


class ZoneCheckInline(admin.TabularInline):
    model = ZoneCheck
    extra = 0
    readonly_fields = ["status", "change_percent", "similarity", "regions", "aligned", "detector"]


@admin.register(Inspection)
class InspectionAdmin(admin.ModelAdmin):
    list_display = ["id", "vehicle", "plate_number", "customer_name", "overall_status", "created_at"]
    list_filter = ["overall_status"]
    search_fields = ["vehicle", "plate_number", "customer_name"]
    inlines = [ZoneCheckInline]


@admin.register(Detection)
class DetectionAdmin(admin.ModelAdmin):
    list_display = ["id", "zone", "label", "confidence", "area_percent", "is_new"]
    list_filter = ["label", "is_new"]
