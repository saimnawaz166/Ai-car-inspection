"""
Hugging Face wali CarDD copy (FiftyOne format) download karke YOLO segmentation
format mein badlo. Form bharne ki zarurat nahi.

    https://huggingface.co/datasets/harpreetsahota/CarDD
    (2816 photos, 6211 damage masks — CarDD ka original train split)

Usage:
    python training/prepare_cardd_hf.py --out datasets/car_damage

Is copy mein sirf train split hai, is liye script khud 80% train / 10% val / 10% test
bana deti hai (fixed seed, har dafa same split).

License: CarDD sirf research / non-commercial use ke liye hai (CarDD_license.pdf dekhein).
"""
import argparse
import base64
import io
import json
import random
import time
import urllib.request
import zlib
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import cv2
import numpy as np

HF_BASE = "https://huggingface.co/datasets/harpreetsahota/CarDD/resolve/main/"
CLASS_KEYS = ["scratch", "dent", "crack", "broken_lamp", "glass_shatter", "flat_tire"]
CLASS_MAP = {
    "scratch": "scratch", "dent": "dent", "crack": "crack",
    "lamp broken": "broken_lamp", "glass shatter": "glass_shatter", "tire flat": "flat_tire",
}


def fetch(url, dest):
    if dest.exists() and dest.stat().st_size > 0:
        return
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(6):  # internet ki chhoti si gadbad par dobara koshish
        try:
            with urllib.request.urlopen(url, timeout=60) as r, open(tmp, "wb") as f:
                f.write(r.read())
            break
        except OSError:
            if attempt == 5:
                raise
            time.sleep(2 ** attempt)
    tmp.replace(dest)


def decode_mask(det):
    """FiftyOne mask: base64 -> zlib -> np.save array (bbox ke andar ka bool mask)."""
    raw = zlib.decompress(base64.b64decode(det["mask"]["$binary"]["base64"]))
    return np.load(io.BytesIO(raw), allow_pickle=False)


def yolo_lines(sample):
    """Har damage ke mask ko polygon(s) mein badlo: 'cls x1 y1 x2 y2 ...' (0-1)."""
    W, H = sample["metadata"]["width"], sample["metadata"]["height"]
    lines = []
    for det in (sample.get("segmentations") or {}).get("detections", []):
        key = CLASS_MAP.get(det["label"].strip().lower())
        if key is None or "mask" not in det:
            continue
        cls = CLASS_KEYS.index(key)
        bx, by, bw, bh = det["bounding_box"]  # normalized [x, y, w, h]
        mask = decode_mask(det).astype(np.uint8)
        mh, mw = mask.shape
        x0, y0 = bx * W, by * H
        sx, sy = (bw * W) / mw, (bh * H) / mh

        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for c in contours:
            if cv2.contourArea(c) < 4:
                continue
            c = cv2.approxPolyDP(c, 1.0, True).reshape(-1, 2).astype(float)
            if len(c) < 3:
                continue
            xs = np.clip((x0 + (c[:, 0] + 0.5) * sx) / W, 0, 1)
            ys = np.clip((y0 + (c[:, 1] + 0.5) * sy) / H, 0, 1)
            pts = " ".join(f"{x:.5f} {y:.5f}" for x, y in zip(xs, ys))
            lines.append(f"{cls} {pts}")
    return lines


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=Path("datasets/car_damage"), type=Path)
    parser.add_argument("--workers", type=int, default=16)
    args = parser.parse_args()
    out = args.out
    raw = out / "_raw"
    raw.mkdir(parents=True, exist_ok=True)

    print("Annotations download ho rahi hain...")
    fetch(HF_BASE + "samples.json", raw / "samples.json")
    fetch(HF_BASE + "CarDD_license.pdf", out / "CarDD_license.pdf")
    samples = json.loads((raw / "samples.json").read_text(encoding="utf-8"))["samples"]

    print(f"{len(samples)} photos download ho rahi hain...")
    img_raw = raw / "images"
    img_raw.mkdir(exist_ok=True)

    def get(s):
        name = Path(s["filepath"]).name
        fetch(HF_BASE + s["filepath"], img_raw / name)

    done = 0
    with ThreadPoolExecutor(args.workers) as pool:
        for _ in pool.map(get, samples):
            done += 1
            if done % 200 == 0:
                print(f"  {done}/{len(samples)}")

    rng = random.Random(42)
    order = list(range(len(samples)))
    rng.shuffle(order)
    n_val = n_test = len(samples) // 10
    split_of = {}
    for rank, i in enumerate(order):
        split_of[i] = "val" if rank < n_val else "test" if rank < n_val + n_test else "train"

    counts = {"train": 0, "val": 0, "test": 0}
    for i, s in enumerate(samples):
        split = split_of[i]
        name = Path(s["filepath"]).name
        (out / "images" / split).mkdir(parents=True, exist_ok=True)
        (out / "labels" / split).mkdir(parents=True, exist_ok=True)
        dest = out / "images" / split / name
        if not dest.exists():
            (img_raw / name).replace(dest)
        (out / "labels" / split / f"{Path(name).stem}.txt").write_text(
            "\n".join(yolo_lines(s)), encoding="utf-8")
        counts[split] += 1

    print(f"Done -> {out.resolve()}  {counts}")


if __name__ == "__main__":
    main()
