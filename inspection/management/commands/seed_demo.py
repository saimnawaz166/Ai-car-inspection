"""
Testing ke liye random demo inspections banao (asli AI model ke saath).

    python manage.py seed_demo              # 8 inspections
    python manage.py seed_demo --count 15
    python manage.py seed_demo --clear      # sirf demo inspections delete karo

Photos kahan se aati hain:
  1. CarDD test photos (datasets/car_damage) — "After" = asli damaged photo,
     "Before" = wohi photo jis mein damage ko inpainting se mita diya gaya.
     Is tarah AI ko "naya damage" milta hai (scratch, dent, lamp, glass, tyre...).
  2. sample_images/ — banaye hue panels (naya scratch / dent).
  3. media/before — asli Corolla photos, Before = After (koi damage nahi).

Demo inspections ke notes mein "[demo]" likha hota hai, taa-ke --clear sirf unhein mitaye.
"""
import random
from datetime import date, timedelta
from pathlib import Path

import cv2
import numpy as np
from django.conf import settings
from django.core.files.base import ContentFile
from django.core.management.base import BaseCommand
from django.db import transaction

from inspection.models import Inspection, ZoneCheck
from inspection.views import run_ai_check

DEMO_TAG = "[demo]"
CLASS_KEYS = ["scratch", "dent", "crack", "broken_lamp", "glass_shatter", "flat_tire"]

# Damage ki class ke hisaab se kis side (view) ki photo mein lagani hai
CLASS_ZONES = {
    "scratch": ["front", "back", "left", "right"],
    "dent": ["front", "back", "left", "right"],
    "crack": ["front", "back"],
    "broken_lamp": ["front", "back"],
    "glass_shatter": ["front", "back"],
    "flat_tire": ["left", "right"],
}
SAMPLE_VIEW = {"front_bumper": "front", "rear_bumper": "back", "left_door": "left"}

VEHICLES = [
    "Toyota Corolla Altis 2022", "Honda Civic RS 2023", "Toyota Fortuner Legender 2024",
    "Suzuki Cultus VXL 2021", "Honda City Aspire 2022", "Toyota Yaris ATIV 2023",
    "Hyundai Tucson AWD 2023", "Kia Sportage Alpha 2024", "Toyota Land Cruiser V8 2021",
    "Mercedes-Benz C200 2022", "Audi A6 2021", "Toyota Prado TX 2020", "Changan Alsvin 2023",
    "MG HS Essence 2024", "Toyota Hilux Revo 2022", "BMW 520i 2022",
]
CUSTOMERS = [
    "Ahmed Raza", "Fatima Khan", "Usman Tariq", "Ayesha Siddiqui", "Bilal Hussain", "Zainab Ali",
    "Hamza Malik", "Sana Javed", "Omar Farooq", "Hira Sheikh", "Ali Haider", "Maryam Qureshi",
    "Saad Iqbal", "Noor Fatima", "Fahad Mustafa", "Iqra Aziz",
]
INSPECTORS = ["Hamid Ali", "Anas Ahmad", "Kashif Mehmood", "Rizwan Saeed", ""]
CITY_CODES = ["LEA", "LEB", "LEC", "LED", "ICT", "BKL", "AFR", "KHI", "RIK", "LEF"]
NOTES = [
    "", "", "Customer ne bataya tha ke parking mein lagi.", "Fuel full return.",
    "Late return — 2 ghante.", "Airport drop.", "Outstation trip (Murree).", "Car wash ke baad photos.",
]


def _plate(rng):
    return f"{rng.choice(CITY_CODES)}-{rng.randint(10, 99) if rng.random() < .3 else ''}{rng.randint(100, 9999)}".replace("--", "-")


def _jpeg(img):
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return ContentFile(buf.tobytes())


def _read(path):
    data = np.fromfile(str(path), dtype=np.uint8)  # Windows paths ke liye safe
    return cv2.imdecode(data, cv2.IMREAD_COLOR)


def _erase_damage(img, label_file):
    """YOLO polygon labels se damage ka mask banao aur inpainting se mita do (= 'Before')."""
    h, w = img.shape[:2]
    mask = np.zeros((h, w), np.uint8)
    for line in label_file.read_text(encoding="utf-8").splitlines():
        parts = line.split()
        if len(parts) < 7:
            continue
        pts = (np.array(parts[1:], float).reshape(-1, 2) * [w, h]).astype(np.int32)
        cv2.fillPoly(mask, [pts], 255)
    mask = cv2.dilate(mask, np.ones((15, 15), np.uint8))
    return cv2.inpaint(img, mask, 9, cv2.INPAINT_TELEA)


class Command(BaseCommand):
    help = "Testing ke liye random demo inspections banata hai (asli AI ke saath)."

    def add_arguments(self, parser):
        parser.add_argument("--count", type=int, default=8)
        parser.add_argument("--seed", type=int, default=None, help="same data dobara chahiye to fixed seed")
        parser.add_argument("--clear", action="store_true", help="sirf demo inspections delete karo")

    def handle(self, *args, count, seed, clear, **opts):
        if clear:
            self._clear()
            return
        rng = random.Random(seed)
        base = Path(settings.BASE_DIR)

        damaged = self._cardd_pool(base)
        synthetic = [(base / "sample_images" / f"{z}_before.jpg", base / "sample_images" / f"{z}_after.jpg",
                      SAMPLE_VIEW[z])
                     for z in ("front_bumper", "rear_bumper", "left_door")
                     if (base / "sample_images" / f"{z}_before.jpg").exists()]
        clean = sorted((base / "media" / "before").glob("*.jp*g"))
        if not (damaged or synthetic or clean):
            self.stderr.write("Koi photo source nahi mila (datasets/, sample_images/, media/before).")
            return
        self.stdout.write(f"Sources: {sum(len(v) for v in damaged.values())} CarDD damage photos, "
                          f"{len(synthetic)} sample pairs, {len(clean)} clean photos")

        for n in range(1, count + 1):
            start = date.today() - timedelta(days=rng.randint(2, 60))
            insp = Inspection.objects.create(
                vehicle=rng.choice(VEHICLES), plate_number=_plate(rng),
                customer_name=rng.choice(CUSTOMERS), inspector_name=rng.choice(INSPECTORS),
                rental_start=start, rental_end=start + timedelta(days=rng.randint(1, 10)),
                notes=f"{rng.choice(NOTES)} {DEMO_TAG}".strip(),
            )
            used = set()
            pairs = self._pick_pairs(rng, damaged, synthetic, clean)
            for zone, before, after in pairs:
                if zone in used:
                    continue
                used.add(zone)
                with transaction.atomic():
                    z = ZoneCheck(inspection=insp, zone=zone)
                    tag = f"demo{insp.pk}_{zone}"
                    z.before_image.save(f"{tag}_before.jpg", _jpeg(before), save=False)
                    z.after_image.save(f"{tag}_after.jpg", _jpeg(after), save=False)
                    z.save()
                run_ai_check(z)
            insp.refresh_overall_status()
            dets = sum(z.detections.filter(is_new=True).count() for z in insp.zones.all())
            self.stdout.write(f"[{n}/{count}] {insp.report_number}  {insp.vehicle:<32} "
                              f"{len(used)} zones  {dets} new damage  -> {insp.get_overall_status_display()}")

        self.stdout.write(self.style.SUCCESS(f"Done. {count} demo inspections bani. "
                                             "Hatane ke liye: python manage.py seed_demo --clear"))

    # ------------------------------------------------------------------
    def _cardd_pool(self, base):
        """CarDD test photos ko unki main damage class ke hisaab se group karo."""
        img_dir = base / "datasets" / "car_damage" / "images" / "test"
        lbl_dir = base / "datasets" / "car_damage" / "labels" / "test"
        pool = {k: [] for k in CLASS_KEYS}
        if not img_dir.exists():
            return {}
        for img in img_dir.glob("*.jpg"):
            lbl = lbl_dir / f"{img.stem}.txt"
            if not lbl.exists():
                continue
            classes = [int(l.split()[0]) for l in lbl.read_text(encoding="utf-8").splitlines() if l.strip()]
            if classes:
                pool[CLASS_KEYS[max(set(classes), key=classes.count)]].append((img, lbl))
        return {k: v for k, v in pool.items() if v}

    def _pick_pairs(self, rng, damaged, synthetic, clean):
        """Ek inspection ke liye 1-4 views: kuch damaged, kuch clean (ek view ek dafa)."""
        pairs = []
        n_damage = rng.choices([0, 1, 2, 3], weights=[2, 4, 3, 1])[0]
        for _ in range(n_damage):
            if damaged and (not synthetic or rng.random() < .8):
                cls = rng.choice(list(damaged))
                img_path, lbl = rng.choice(damaged[cls])
                after = _read(img_path)
                pairs.append((rng.choice(CLASS_ZONES[cls]), _erase_damage(after, lbl), after))
            elif synthetic:
                b, a, zone = rng.choice(synthetic)
                pairs.append((zone, _read(b), _read(a)))
        clean_zones = ["front", "back", "left", "right"]
        for _ in range(rng.randint(1, 2) if clean else 0):
            img = _read(rng.choice(clean))
            pairs.append((rng.choice(clean_zones), img, img.copy()))
        rng.shuffle(pairs)
        return pairs

    def _clear(self):
        qs = Inspection.objects.filter(notes__contains=DEMO_TAG)
        n = qs.count()
        for insp in qs:
            for z in insp.zones.all():
                for f in (z.before_image, z.after_image, z.annotated_image):
                    if f:
                        f.delete(save=False)
            insp.delete()
        self.stdout.write(self.style.SUCCESS(f"{n} demo inspections delete ho gayi."))
