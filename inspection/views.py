import uuid
from pathlib import Path

from django.conf import settings
from django.contrib import messages
from django.db import transaction
from django.http import HttpResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.views.decorators.http import require_POST

from . import ai_engine, damage_detector
from .forms import InspectionForm, collect_zone_files
from .models import VIEW_CHOICES, VIEWS, Detection, Inspection, ZoneCheck
from .report_pdf import build_pdf


def home(request):
    """Saari inspections ki list."""
    inspections = Inspection.objects.prefetch_related("zones")
    return render(request, "inspection/home.html", {"inspections": inspections})


def run_ai_check(zone):
    """Ek zone ki photos AI engine ko do aur result model mein save karo."""
    out_dir = Path(settings.MEDIA_ROOT) / "annotated"
    out_dir.mkdir(parents=True, exist_ok=True)
    filename = f"{zone.zone}_{uuid.uuid4().hex[:10]}.jpg"

    try:
        view = zone.zone if zone.zone in dict(VIEW_CHOICES) else None
        result = ai_engine.compare_images(zone.before_image.path, zone.after_image.path,
                                          out_dir / filename, view=view)
    except Exception:  # kharab/corrupt photo waghera
        zone.status = ai_engine.STATUS_UNCLEAR
        zone.save()
        return

    zone.status = result["status"]
    zone.change_percent = result["change_percent"]
    zone.similarity = result["similarity"]
    zone.regions = result["regions"]
    zone.aligned = result["aligned"]
    zone.annotated_image.name = f"annotated/{filename}"
    zone.detector = result["detector"] or ""
    zone.save()

    Detection.objects.bulk_create([
        Detection(zone=zone, label=d["label"], confidence=d["confidence"],
                  x1=d["bbox"][0], y1=d["bbox"][1], x2=d["bbox"][2], y2=d["bbox"][3],
                  polygon=d["polygon"], area_percent=d["area_percent"], is_new=d["is_new"],
                  part=d.get("part", ""))
        for d in result["detections"]
        if d["label"] in damage_detector.CLASS_LABELS
    ])


def new_inspection(request):
    """Form dikhao (GET) ya photos le kar AI inspection chalao (POST)."""
    form = InspectionForm(request.POST or None)
    zone_errors = []

    if request.method == "POST":
        pairs, zone_errors = collect_zone_files(request.FILES)
        if form.is_valid() and not zone_errors:
            with transaction.atomic():
                inspection = form.save()
                for key, before, after in pairs:
                    zone = ZoneCheck.objects.create(inspection=inspection, zone=key,
                                                    before_image=before, after_image=after)
                    run_ai_check(zone)
                inspection.refresh_overall_status()
            messages.success(request, "Inspection complete. AI report tayyar hai.")
            return redirect("inspection:report", pk=inspection.pk)

    return render(request, "inspection/new.html", {
        "form": form,
        "views": VIEWS,
        "zone_errors": zone_errors,
    })


def report(request, pk):
    """Vehicle Inspection Report (HTML)."""
    inspection = get_object_or_404(Inspection.objects.prefetch_related("zones__detections"), pk=pk)
    return render(request, "inspection/report.html", {"inspection": inspection})


def report_pdf(request, pk):
    """Wohi report PDF mein download."""
    inspection = get_object_or_404(Inspection, pk=pk)
    pdf = build_pdf(inspection)
    response = HttpResponse(pdf, content_type="application/pdf")
    response["Content-Disposition"] = f'attachment; filename="{inspection.report_number}.pdf"'
    return response


@require_POST
def delete_inspection(request, pk):
    inspection = get_object_or_404(Inspection, pk=pk)
    label = f"{inspection.report_number} ({inspection.vehicle})"
    # Photos disk se bhi delete karo
    for z in inspection.zones.all():
        for f in (z.before_image, z.after_image, z.annotated_image):
            if f:
                f.delete(save=False)
    inspection.delete()
    messages.success(request, f"{label} delete ho gayi.")
    return redirect("inspection:home")
