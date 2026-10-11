"""
Trained model ko asli (apni) photos par test karo.

Har photo par damage detection chalata hai aur marked photo
runs/real_test/ mein save karta hai, saath har damage ki list print karta hai.

Usage:
    python training/test_real_photos.py                       # media/before + media/after
    python training/test_real_photos.py --images D:/photos    # koi bhi folder
    python training/test_real_photos.py --weights runs/car_damage/weights/last.pt   # training ke dauran
"""
import argparse
import os
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--images", nargs="*",
                        default=[str(ROOT / "media" / "before"), str(ROOT / "media" / "after")])
    parser.add_argument("--weights", help="default: ml_models/car_damage_yolo11-seg.pt")
    parser.add_argument("--conf", type=float, default=None)
    parser.add_argument("--out", default=str(ROOT / "runs" / "real_test"))
    args = parser.parse_args()

    if args.weights:
        os.environ["DAMAGE_MODEL_PATH"] = str(Path(args.weights).resolve())
    from inspection import damage_detector as dd  # env set hone ke baad import

    if not dd.is_available():
        raise SystemExit(dd.status_message())
    conf = args.conf if args.conf is not None else dd.CONF_THRESHOLD

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    files = []
    for folder in args.images:
        p = Path(folder)
        files += sorted(f for f in (p.iterdir() if p.is_dir() else [p])
                        if f.suffix.lower() in {".jpg", ".jpeg", ".png", ".webp"})

    print(f"Model: {dd.model_path()}  conf>={conf}  photos: {len(files)}\n")
    total = 0
    for f in files:
        img = cv2.imread(str(f))
        if img is None:
            print(f"[skip] {f.name}")
            continue
        dets = dd.detect(img, conf=conf)
        total += len(dets)
        name = f"{f.parent.name}_{f.stem}.jpg"
        cv2.imwrite(str(out / name), dd.draw_detections(img, dets))
        found = ", ".join(f"{d['label']} {d['confidence']:.2f}" for d in dets) or "no damage"
        print(f"{f.parent.name}/{f.name}: {found}")

    print(f"\nTotal detections: {total}.  Marked photos -> {out}")


if __name__ == "__main__":
    main()
