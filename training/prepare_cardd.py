"""
CarDD (COCO format) dataset ko YOLO segmentation format mein badlo.

CarDD ki 6 classes bilkul hamari wali hain:
    dent, scratch, crack, glass shatter, lamp broken, tire flat

Usage:
    python training/prepare_cardd.py --coco-root path/to/CarDD_COCO --out datasets/car_damage

CarDD_COCO folder ke andar ye hona chahiye:
    annotations/instances_train2017.json   train2017/
    annotations/instances_val2017.json     val2017/
    annotations/instances_test2017.json    test2017/

Apna dataset (Roboflow / CVAT se COCO export) bhi isi se convert ho sakta hai,
bas class ke naam neeche CLASS_MAP mein hone chahiye.
"""
import argparse
import json
import shutil
from collections import defaultdict
from pathlib import Path

CLASS_KEYS = ["scratch", "dent", "crack", "broken_lamp", "glass_shatter", "flat_tire"]
CLASS_MAP = {
    "scratch": "scratch",
    "dent": "dent",
    "crack": "crack",
    "lamp broken": "broken_lamp", "broken lamp": "broken_lamp", "broken_lamp": "broken_lamp",
    "glass shatter": "glass_shatter", "glass_shatter": "glass_shatter",
    "tire flat": "flat_tire", "flat tire": "flat_tire", "flat_tire": "flat_tire",
}
SPLITS = {"train": "train2017", "val": "val2017", "test": "test2017"}


def convert_split(coco_root, out, split, folder):
    ann_file = coco_root / "annotations" / f"instances_{folder}.json"
    if not ann_file.exists():
        print(f"[skip] {ann_file} nahi mili")
        return
    data = json.loads(ann_file.read_text(encoding="utf-8"))

    cat_to_cls = {}
    for c in data["categories"]:
        key = CLASS_MAP.get(c["name"].strip().lower())
        if key is None:
            print(f"[warn] class '{c['name']}' ignore ki")
            continue
        cat_to_cls[c["id"]] = CLASS_KEYS.index(key)

    anns = defaultdict(list)
    for a in data["annotations"]:
        anns[a["image_id"]].append(a)

    img_dir = out / "images" / split
    lbl_dir = out / "labels" / split
    img_dir.mkdir(parents=True, exist_ok=True)
    lbl_dir.mkdir(parents=True, exist_ok=True)

    n_obj = 0
    for img in data["images"]:
        w, h = img["width"], img["height"]
        lines = []
        for a in anns[img["id"]]:
            cls = cat_to_cls.get(a["category_id"])
            seg = a.get("segmentation")
            if cls is None or a.get("iscrowd") or not isinstance(seg, list):
                continue  # RLE / crowd masks skip
            for poly in seg:  # ek damage ke kai tukre ho sakte hain
                if len(poly) < 6:
                    continue
                pts = [f"{min(max(v / (w if i % 2 == 0 else h), 0), 1):.6f}"
                       for i, v in enumerate(poly)]
                lines.append(f"{cls} " + " ".join(pts))
                n_obj += 1
        src = coco_root / folder / img["file_name"]
        if not src.exists():
            continue
        shutil.copy2(src, img_dir / src.name)
        (lbl_dir / f"{Path(img['file_name']).stem}.txt").write_text("\n".join(lines), encoding="utf-8")

    print(f"{split}: {len(data['images'])} images, {n_obj} damage polygons")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--coco-root", required=True, type=Path)
    parser.add_argument("--out", default=Path("datasets/car_damage"), type=Path)
    args = parser.parse_args()
    for split, folder in SPLITS.items():
        convert_split(args.coco_root, args.out, split, folder)
    print(f"Done -> {args.out.resolve()}")


if __name__ == "__main__":
    main()
