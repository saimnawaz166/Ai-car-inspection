import django.db.models.deletion
from django.db import migrations, models

STATUS = [
    ("no_change", "No significant change"),
    ("minor", "Possible new scratch / minor change"),
    ("damage", "Possible damage"),
    ("unclear", "Unclear - please retake photo from the same angle"),
]
ZONES = [
    ("front_bumper", "Front Bumper"),
    ("rear_bumper", "Rear Bumper"),
    ("left_door", "Left Door"),
    ("right_door", "Right Door"),
    ("hood", "Hood"),
    ("trunk", "Trunk / Boot"),
]


class Migration(migrations.Migration):

    initial = True
    dependencies = []

    operations = [
        migrations.CreateModel(
            name="Inspection",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("vehicle", models.CharField(max_length=120, verbose_name="Vehicle (make / model)")),
                ("plate_number", models.CharField(max_length=30)),
                ("customer_name", models.CharField(max_length=120)),
                ("inspector_name", models.CharField(blank=True, max_length=120)),
                ("rental_start", models.DateField(blank=True, null=True)),
                ("rental_end", models.DateField(blank=True, null=True)),
                ("notes", models.TextField(blank=True)),
                ("overall_status", models.CharField(choices=STATUS, default="no_change", max_length=20)),
                ("created_at", models.DateTimeField(auto_now_add=True)),
            ],
            options={"ordering": ["-created_at"]},
        ),
        migrations.CreateModel(
            name="ZoneCheck",
            fields=[
                ("id", models.BigAutoField(auto_created=True, primary_key=True, serialize=False, verbose_name="ID")),
                ("zone", models.CharField(choices=ZONES, max_length=30)),
                ("before_image", models.ImageField(upload_to="before/")),
                ("after_image", models.ImageField(upload_to="after/")),
                ("annotated_image", models.ImageField(blank=True, upload_to="annotated/")),
                ("status", models.CharField(choices=STATUS, default="no_change", max_length=20)),
                ("change_percent", models.FloatField(default=0)),
                ("similarity", models.FloatField(default=0)),
                ("regions", models.PositiveIntegerField(default=0)),
                ("aligned", models.BooleanField(default=True)),
                ("inspection", models.ForeignKey(on_delete=django.db.models.deletion.CASCADE,
                                                 related_name="zones", to="inspection.inspection")),
            ],
            options={"ordering": ["id"]},
        ),
    ]
