from django.db import models

from . import ai_engine, damage_detector

STATUS_CHOICES = [(key, label) for key, label in ai_engine.STATUS_LABELS.items()]

# Har inspection mein sirf 4 photos: poori gaadi aage, peeche, left aur right se.
# AI har photo mein saare hisse (shishe, lights, tyre, bumper, darwaze) khud check karta hai.
VIEWS = [
    ("front", "Front", "Saamne se, 2-3 meter door — poori gaadi frame mein"),
    ("back", "Back", "Peeche se, 2-3 meter door — poori gaadi frame mein"),
    ("left", "Left side", "Left side se, aage se peeche tak poori gaadi"),
    ("right", "Right side", "Right side se, aage se peeche tak poori gaadi"),
]
VIEW_CHOICES = [(key, label) for key, label, _ in VIEWS]

# Purani inspections (14-zone wala tareeqa) ke naam — sirf purani reports dikhane ke liye
LEGACY_ZONE_CHOICES = [
    ("front_bumper", "Front Bumper"), ("rear_bumper", "Rear Bumper"),
    ("left_door", "Left Door"), ("right_door", "Right Door"),
    ("hood", "Hood"), ("trunk", "Trunk / Boot"),
    ("windshield", "Windshield"), ("rear_glass", "Rear Glass"),
    ("headlights", "Headlights"), ("taillights", "Taillights"),
    ("tyre_front_left", "Tyre - Front Left"), ("tyre_front_right", "Tyre - Front Right"),
    ("tyre_rear_left", "Tyre - Rear Left"), ("tyre_rear_right", "Tyre - Rear Right"),
]
ZONE_CHOICES = VIEW_CHOICES + LEGACY_ZONE_CHOICES


class Inspection(models.Model):
    """Ek rental ki poori inspection (car ki details + overall result)."""

    vehicle = models.CharField("Vehicle (make / model)", max_length=120)
    plate_number = models.CharField(max_length=30)
    customer_name = models.CharField(max_length=120)
    inspector_name = models.CharField(max_length=120, blank=True)
    rental_start = models.DateField(null=True, blank=True)
    rental_end = models.DateField(null=True, blank=True)
    notes = models.TextField(blank=True)

    overall_status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="no_change")
    created_at = models.DateTimeField(auto_now_add=True)

    class Meta:
        ordering = ["-created_at"]

    def __str__(self):
        return f"#{self.pk} {self.vehicle} ({self.plate_number})"

    @property
    def report_number(self):
        return f"VIR-{self.created_at:%Y%m%d}-{self.pk:04d}"

    def refresh_overall_status(self):
        statuses = list(self.zones.values_list("status", flat=True))
        self.overall_status = ai_engine.overall_status(statuses)
        self.save(update_fields=["overall_status"])


class ZoneCheck(models.Model):
    """Car ke ek hisse (zone) ki before/after photo aur AI ka result."""

    inspection = models.ForeignKey(Inspection, on_delete=models.CASCADE, related_name="zones")
    zone = models.CharField("View / zone", max_length=30, choices=ZONE_CHOICES)

    before_image = models.ImageField(upload_to="before/")
    after_image = models.ImageField(upload_to="after/")
    annotated_image = models.ImageField(upload_to="annotated/", blank=True)

    status = models.CharField(max_length=20, choices=STATUS_CHOICES, default="no_change")
    change_percent = models.FloatField(default=0)
    similarity = models.FloatField(default=0)
    regions = models.PositiveIntegerField(default=0)
    aligned = models.BooleanField(default=True)
    detector = models.CharField("AI model used", max_length=120, blank=True)

    class Meta:
        ordering = ["id"]

    def __str__(self):
        return f"{self.get_zone_display()} - {self.get_status_display()}"

    @property
    def new_detections(self):
        return [d for d in self.detections.all() if d.is_new]


DAMAGE_CHOICES = [(key, damage_detector.CLASS_LABELS[key]) for key in damage_detector.CLASS_KEYS]


class Detection(models.Model):
    """
    YOLO11-seg ka mila hua ek damage (After photo par).
    Box aur polygon annotated image ke pixel coordinates mein hain.
    """

    zone = models.ForeignKey(ZoneCheck, on_delete=models.CASCADE, related_name="detections")
    label = models.CharField(max_length=30, choices=DAMAGE_CHOICES)
    confidence = models.FloatField()
    x1 = models.IntegerField()
    y1 = models.IntegerField()
    x2 = models.IntegerField()
    y2 = models.IntegerField()
    polygon = models.JSONField(default=list, help_text="Mask outline: [[x, y], ...]")
    area_percent = models.FloatField(default=0)
    part = models.CharField("Location (approx.)", max_length=60, blank=True)
    is_new = models.BooleanField(default=True, help_text="False = Before photo mein pehle se tha")

    class Meta:
        ordering = ["-is_new", "-confidence"]

    def __str__(self):
        return f"{self.get_label_display()} ({self.confidence:.0%})"

    @property
    def bbox(self):
        return [self.x1, self.y1, self.x2, self.y2]
