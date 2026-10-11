from django import forms

from .models import VIEW_CHOICES, Inspection


class InspectionForm(forms.ModelForm):
    """Car aur customer ki details ka form. Photos alag se (har zone ke liye) aati hain."""

    class Meta:
        model = Inspection
        fields = ["vehicle", "plate_number", "customer_name", "inspector_name",
                  "rental_start", "rental_end", "notes"]
        widgets = {
            "vehicle": forms.TextInput(attrs={"placeholder": "e.g. Toyota Corolla 2022"}),
            "plate_number": forms.TextInput(attrs={"placeholder": "e.g. LEA-1234"}),
            "customer_name": forms.TextInput(attrs={"placeholder": "Renter ka naam"}),
            "inspector_name": forms.TextInput(attrs={"placeholder": "Staff member"}),
            "rental_start": forms.DateInput(attrs={"type": "date"}),
            "rental_end": forms.DateInput(attrs={"type": "date"}),
            "notes": forms.Textarea(attrs={"rows": 2, "placeholder": "Optional notes"}),
        }

    def clean(self):
        data = super().clean()
        start, end = data.get("rental_start"), data.get("rental_end")
        if start and end and end < start:
            self.add_error("rental_end", "Rental end date, start date se pehle nahi ho sakti.")
        return data


ALLOWED_TYPES = {"image/jpeg", "image/png", "image/webp"}


def collect_zone_files(files):
    """
    request.FILES se har view (front/back/left/right) ki before/after photo uthao.

    Returns: (pairs, errors)
        pairs  -> [(zone_key, before_file, after_file), ...]
        errors -> list of error messages
    """
    pairs, errors = [], []
    for key, label in VIEW_CHOICES:
        before = files.get(f"before_{key}")
        after = files.get(f"after_{key}")
        if not before and not after:
            continue  # yeh view skip kiya gaya
        if not (before and after):
            errors.append(f"{label}: Before aur After dono photos zaroori hain.")
            continue
        bad = [f.name for f in (before, after) if f.content_type not in ALLOWED_TYPES]
        if bad:
            errors.append(f"{label}: sirf JPG/PNG/WEBP photos upload karein ({', '.join(bad)}).")
            continue
        pairs.append((key, before, after))

    if not pairs and not errors:
        errors.append("Kam az kam ek side (Front / Back / Left / Right) ki Before aur After photo upload karein.")
    return pairs, errors
