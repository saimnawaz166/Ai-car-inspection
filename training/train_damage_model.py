"""
YOLO11-seg ko car damage par train karo.

Usage:
    python training/train_damage_model.py                  # default: yolo11s-seg, 100 epochs
    python training/train_damage_model.py --model yolo11m-seg.pt --epochs 150   # zyada accuracy
    python training/train_damage_model.py --device cpu --batch 4                 # GPU na ho

Model sizes: yolo11n-seg (sab se tez) < s < m < l < x (sab se accurate, slow).
GPU na ho to Google Colab (free T4) par chalayein.

Training ke baad best weights khud copy ho jate hain:
    ml_models/car_damage_yolo11-seg.pt   <- app yahin se load karti hai
"""
import argparse
import shutil
from pathlib import Path

import yaml
from ultralytics import YOLO

ROOT = Path(__file__).resolve().parent.parent


def resolve_data_yaml(data):
    """data.yaml ka relative 'path' project folder ke hisaab se absolute karo."""
    data = Path(data).resolve()
    cfg = yaml.safe_load(data.read_text(encoding="utf-8"))
    if not Path(cfg["path"]).is_absolute():
        cfg["path"] = str((data.parent / cfg["path"]).resolve())
    out = ROOT / "runs" / "data_resolved.yaml"
    out.parent.mkdir(exist_ok=True)
    out.write_text(yaml.safe_dump(cfg, sort_keys=False), encoding="utf-8")
    return str(out)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="yolo11s-seg.pt", help="pretrained COCO weights se start")
    parser.add_argument("--data", default=str(ROOT / "training" / "data.yaml"))
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--batch", type=int, default=16)
    parser.add_argument("--device", default=None, help="0 (GPU) ya cpu")
    parser.add_argument("--workers", type=int, default=4, help="data loading threads")
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--resume", action="store_true", help="ruki hui training wahin se chalao")
    args = parser.parse_args()
    data = resolve_data_yaml(args.data)

    if args.resume:
        model = YOLO(str(ROOT / "runs" / "car_damage" / "weights" / "last.pt"))
        model.train(resume=True)
    else:
        model = YOLO(args.model)
        model.train(
            data=data,
            epochs=args.epochs,
            imgsz=args.imgsz,
            batch=args.batch,
            device=args.device,
            patience=args.patience,  # itne epochs tak behtari na ho to ruk jao
            workers=args.workers,
            project=str(ROOT / "runs"),
            name="car_damage",
            exist_ok=True,
            # Car photos ke liye augmentation: ulta (upside down) nahi, left-right theek hai
            fliplr=0.5,
            flipud=0.0,
            degrees=5.0,
            hsv_v=0.4,             # roshni ka farq
        )

    best = ROOT / "runs" / "car_damage" / "weights" / "best.pt"
    model = YOLO(str(best))
    metrics = model.val(data=data, split="test")
    print(f"Test mask mAP50: {metrics.seg.map50:.3f}   mAP50-95: {metrics.seg.map:.3f}")
    print(f"Test box  mAP50: {metrics.box.map50:.3f}   mAP50-95: {metrics.box.map:.3f}")

    target = ROOT / "ml_models" / "car_damage_yolo11-seg.pt"
    target.parent.mkdir(exist_ok=True)
    shutil.copy2(best, target)
    print(f"Model saved -> {target}")


if __name__ == "__main__":
    main()
