"""
YOLO11-seg damage detector (instance segmentation).

Kaam: ek photo mein car ke damage dhoondna aur har damage ka
  - class      (scratch, dent, crack, broken_lamp, glass_shatter, flat_tire)
  - confidence (0-1)
  - bounding box (x1, y1, x2, y2 pixels)
  - mask       (polygon + binary mask)
wapas dena.

Model file (trained weights) default yahan dhoondi jati hai:
    ml_models/car_damage_yolo11-seg.pt
Ya environment variable se path dein:  DAMAGE_MODEL_PATH=C:\\path\\best.pt

Agar `ultralytics` install nahi ya model file nahi mili to detector
"unavailable" rehta hai aur app purane SSIM comparison par chalti rehti hai.

Model train karne ka tareeqa: training/README.md

Command line test (Django ki zarurat nahi):
    python -m inspection.damage_detector photo.jpg [--save out.jpg]
"""

import os
import threading
from pathlib import Path

import cv2
import numpy as np

BASE_DIR = Path(__file__).resolve().parent.parent
DEFAULT_MODEL_PATH = BASE_DIR / "ml_models" / "car_damage_yolo11-seg.pt"
# COCO model (car / bus / truck) — photo mein gaadi ka outline nikalne ke liye
CAR_FINDER_PATH = BASE_DIR / "ml_models" / "car_finder_yolo11n-seg.pt"
VEHICLE_CLASSES = [2, 5, 7]  # COCO: car, bus, truck

CONF_THRESHOLD = float(os.environ.get("DAMAGE_MODEL_CONF", 0.35))
IOU_THRESHOLD = 0.5
# Poori gaadi ki photo mein damage chhota hota hai, is liye 640 se bari input size
IMG_SIZE = int(os.environ.get("DAMAGE_MODEL_IMGSZ", 960))

# Hamari 6 classes (training/data.yaml mein bhi yahi order hai)
CLASS_KEYS = ["scratch", "dent", "crack", "broken_lamp", "glass_shatter", "flat_tire"]
CLASS_LABELS = {
    "scratch": "Scratch",
    "dent": "Dent",
    "crack": "Crack",
    "broken_lamp": "Broken lamp",
    "glass_shatter": "Glass shatter",
    "flat_tire": "Flat tire",
}
# Doosre datasets (e.g. CarDD) mein naam thore alag hote hain
_ALIASES = {
    "lamp_broken": "broken_lamp",
    "lamp": "broken_lamp",
    "tire_flat": "flat_tire",
    "flat_tyre": "flat_tire",
    "glass": "glass_shatter",
    "shattered_glass": "glass_shatter",
}

# Annotated photo par har class ka rang (BGR)
CLASS_COLORS = {
    "scratch": (0, 165, 255),      # orange
    "dent": (0, 0, 230),           # red
    "crack": (180, 0, 200),        # purple
    "broken_lamp": (0, 220, 255),  # yellow
    "glass_shatter": (255, 160, 0),  # blue
    "flat_tire": (60, 60, 60),     # dark gray
}

_model = None
_model_error = None
_car_finder = None
_car_finder_loaded = False
_lock = threading.Lock()


def normalize_class(name):
    """Model ka class naam hamari standard key mein badlo."""
    key = str(name).strip().lower().replace(" ", "_").replace("-", "_")
    return _ALIASES.get(key, key)


def model_path():
    return Path(os.environ.get("DAMAGE_MODEL_PATH", DEFAULT_MODEL_PATH))


def get_model():
    """Model sirf ek dafa load karo (pehli request par). Na mile to None."""
    global _model, _model_error
    if _model is not None or _model_error is not None:
        return _model
    with _lock:
        if _model is not None or _model_error is not None:
            return _model
        path = model_path()
        if not path.exists():
            _model_error = f"Model file nahi mili: {path}"
            return None
        try:
            from ultralytics import YOLO
        except ImportError:
            _model_error = "ultralytics install nahi hai (pip install ultralytics)"
            return None
        _model = YOLO(str(path))
    return _model


def is_available():
    return get_model() is not None


def status_message():
    """Report/admin mein dikhane ke liye: model ka haal."""
    if is_available():
        return f"YOLO11-seg ({model_path().name})"
    return _model_error or "Model not loaded"


def detect(image, conf=CONF_THRESHOLD):
    """
    Photo mein damage dhoondo.

    image: file path ya BGR numpy array (OpenCV).

    Returns list of dicts (confidence ke hisaab se sorted):
        label         -> 'scratch' | 'dent' | ...
        confidence    -> 0-1
        bbox          -> [x1, y1, x2, y2] pixels
        polygon       -> [[x, y], ...] pixels (mask ka outline)
        mask          -> HxW bool numpy array (photo ke size ka)
        area_percent  -> photo ka kitna % hissa ye damage hai
    Model na ho to khaali list.
    """
    model = get_model()
    if model is None:
        return []

    if not isinstance(image, np.ndarray):
        image = cv2.imread(str(image))
        if image is None:
            raise ValueError("Image read nahi ho saki")
    h, w = image.shape[:2]

    # retina_masks=True -> masks seedha original photo ke size mein
    result = model.predict(image, conf=conf, iou=IOU_THRESHOLD, imgsz=IMG_SIZE,
                           retina_masks=True, verbose=False)[0]
    if result.boxes is None or len(result.boxes) == 0:
        return []

    boxes = result.boxes.xyxy.cpu().numpy()
    confs = result.boxes.conf.cpu().numpy()
    classes = result.boxes.cls.cpu().numpy().astype(int)
    masks = result.masks.data.cpu().numpy() if result.masks is not None else None
    polygons = result.masks.xy if result.masks is not None else None

    detections = []
    for i in range(len(boxes)):
        label = normalize_class(result.names[classes[i]])
        if masks is not None:
            mask = masks[i] > 0.5
            if mask.shape != (h, w):
                mask = cv2.resize(mask.astype(np.uint8), (w, h),
                                  interpolation=cv2.INTER_NEAREST).astype(bool)
            polygon = [[round(float(x), 1), round(float(y), 1)] for x, y in polygons[i]]
        else:  # detect-only model: box ko hi mask maan lo
            mask = np.zeros((h, w), dtype=bool)
            x1, y1, x2, y2 = boxes[i].astype(int)
            mask[y1:y2, x1:x2] = True
            polygon = [[x1, y1], [x2, y1], [x2, y2], [x1, y2]]

        detections.append({
            "label": label,
            "confidence": round(float(confs[i]), 3),
            "bbox": [int(round(v)) for v in boxes[i]],
            "polygon": polygon,
            "mask": mask,
            "area_percent": round(100.0 * float(mask.sum()) / (h * w), 3),
        })

    detections.sort(key=lambda d: d["confidence"], reverse=True)
    return detections


def _get_car_finder():
    global _car_finder, _car_finder_loaded
    if _car_finder_loaded:
        return _car_finder
    with _lock:
        if not _car_finder_loaded:
            try:
                from ultralytics import YOLO
                _car_finder = YOLO(str(CAR_FINDER_PATH)) if CAR_FINDER_PATH.exists() else None
            except ImportError:
                _car_finder = None
            _car_finder_loaded = True
    return _car_finder


def car_mask(image):
    """
    Photo mein sab se bari gaadi ka mask (HxW bool). Gaadi na mile ya model na ho to None.
    Isse comparison aur detection sirf gaadi par hote hain, background (log, deewar,
    doosri gaadiyan) par nahi.
    """
    model = _get_car_finder()
    if model is None:
        return None
    h, w = image.shape[:2]
    result = model.predict(image, imgsz=640, conf=0.3, classes=VEHICLE_CLASSES,
                           retina_masks=True, verbose=False)[0]
    if result.masks is None or len(result.boxes) == 0:
        return None
    masks = result.masks.data.cpu().numpy() > 0.5
    mask = masks[int(masks.reshape(len(masks), -1).sum(1).argmax())]
    if mask.shape != (h, w):
        mask = cv2.resize(mask.astype(np.uint8), (w, h), interpolation=cv2.INTER_NEAREST).astype(bool)
    return mask


def locate(label, view, bbox, frame):
    """
    Damage gaadi ke kis hisse par hai — photo ki side (view) + gaadi ke andar position se.
    Andaza hai (approx.), pakki measurement nahi.

    frame: gaadi ka box (x1, y1, x2, y2) photo mein.
    Front photo mein gaadi ki RIGHT side tasveer ke LEFT par hoti hai (aap gaadi ke saamne khare hain).
    Left side photo mein gaadi ka FRONT tasveer ke LEFT par hota hai; right side mein ulta.
    """
    fx1, fy1, fx2, fy2 = frame
    x1, y1, x2, y2 = bbox
    cx = ((x1 + x2) / 2 - fx1) / max(fx2 - fx1, 1)
    cy = ((y1 + y2) / 2 - fy1) / max(fy2 - fy1, 1)

    if view in ("front", "back"):
        front = view == "front"
        if front:
            side = "right" if cx < 0.5 else "left"
        else:
            side = "left" if cx < 0.5 else "right"
        if label == "glass_shatter":
            return "Windshield" if front else "Rear glass"
        if label == "broken_lamp":
            return f"{'Headlight' if front else 'Taillight'} ({side})"
        if label == "flat_tire":
            return f"{'Front' if front else 'Rear'} {side} tyre"
        if cy < 0.45:
            return "Hood" if front else "Trunk lid"
        return f"{'Front' if front else 'Rear'} bumper ({side})"

    if view in ("left", "right"):
        side = "Left" if view == "left" else "Right"
        frontness = 1 - cx if view == "left" else cx  # 1 = gaadi ka agla sira
        end = "front" if frontness > 0.5 else "rear"
        if label == "flat_tire":
            return f"{end.title()} {side.lower()} tyre"
        if label == "broken_lamp":
            return f"{'Headlight' if end == 'front' else 'Taillight'} ({side.lower()})"
        if label == "glass_shatter":
            return f"{side} {end} window"
        if frontness > 0.78:
            return f"{side} front fender"
        if frontness < 0.22:
            return f"{side} rear quarter panel"
        return f"{side} {end} door"
    return ""


def draw_detections(image, detections, alpha=0.4):
    """Masks (rang bhar ke), boxes aur 'label 0.87' photo par banao."""
    out = image.copy()
    overlay = image.copy()
    for d in detections:
        if d.get("is_new") is False:
            continue
        overlay[d["mask"]] = CLASS_COLORS.get(d["label"], (0, 0, 255))
    out = cv2.addWeighted(overlay, alpha, out, 1 - alpha, 0)

    for d in detections:
        x1, y1, x2, y2 = d["bbox"]
        old = d.get("is_new") is False
        color = (150, 150, 150) if old else CLASS_COLORS.get(d["label"], (0, 0, 255))
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 1 if old else 2)
        text = f"{CLASS_LABELS.get(d['label'], d['label'])} {d['confidence']:.2f}"
        if d.get("part"):
            text += f" - {d['part']}"
        if old:
            text += " (old)"
        (tw, th), base = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)
        ty = y1 - 4 if y1 - th - 6 > 0 else y2 + th + 4
        cv2.rectangle(out, (x1, ty - th - 3), (x1 + tw + 4, ty + base - 1), color, cv2.FILLED)
        cv2.putText(out, text, (x1 + 2, ty - 1), cv2.FONT_HERSHEY_SIMPLEX, 0.5,
                    (255, 255, 255), 1, cv2.LINE_AA)
    return out


def to_json(detections):
    """Numpy mask hata kar JSON-safe list (API/DB ke liye)."""
    return [{k: v for k, v in d.items() if k != "mask"} for d in detections]


if __name__ == "__main__":
    import argparse
    import json

    parser = argparse.ArgumentParser(description="Car damage detection (YOLO11-seg)")
    parser.add_argument("image")
    parser.add_argument("--save", help="annotated image yahan save karo")
    parser.add_argument("--conf", type=float, default=CONF_THRESHOLD)
    args = parser.parse_args()

    if not is_available():
        raise SystemExit(status_message())
    img = cv2.imread(args.image)
    found = detect(img, conf=args.conf)
    print(json.dumps(to_json(found), indent=2))
    if args.save:
        cv2.imwrite(args.save, draw_detections(img, found))
        print(f"Saved: {args.save}")
