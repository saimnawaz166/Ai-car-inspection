# YOLO11-seg car damage model — training

Model: **YOLO11-seg** (Ultralytics). Tez hai, CPU par bhi inference chal jata hai,
aur training ek command hai. Output: har damage ka **mask + box + confidence**.

## 0. Sab se aasaan: Google Colab (recommended)

Laptop par training CPU ko ghanton 100% par chalati hai — garmi se laptop band ho sakta hai.
Colab par muft GPU milta hai (~1-2 ghante):

1. https://colab.research.google.com → **Upload** → `training/colab_train.ipynb`
2. **Runtime → Change runtime type → T4 GPU**
3. **Runtime → Run all**
4. Aakhir mein `car_damage_yolo11-seg.pt` download hogi → project ke `ml_models/` folder mein rakhein.

## 1. Install

```bash
pip install ultralytics        # torch bhi saath aata hai
# NVIDIA GPU ho to pehle CUDA wala torch: https://pytorch.org/get-started/locally/
```

## 2. Dataset — CarDD (recommended)

[CarDD](https://cardd-ustc.github.io/) ~4,000 car photos, ~9,000 damage masks, aur iski 6 classes
bilkul hamari hain: dent, scratch, crack, glass shatter, lamp broken, tire flat.
Website par form bhar ke download link milta hai (research/non-commercial license — commercial
use se pehle license check karein).

Form ke baghair: `python training/prepare_cardd_hf.py` Hugging Face wali copy (2,816 photos)
khud download karke convert kar deta hai.

Official download ke baad `CarDD_COCO` folder ko YOLO format mein convert karein:

```bash
python training/prepare_cardd.py --coco-root D:/CarDD_release/CarDD_COCO --out datasets/car_damage
```

Apni rental company ki photos bhi add karein (best results ke liye zaroori): CVAT ya Roboflow
mein polygon se label karein, **COCO** format mein export karein, aur isi script se convert
karke `datasets/car_damage/images|labels/train` mein daal dein. Class naam same rakhein.

## 3. Train

```bash
# GPU (ya Google Colab free T4) — ~1-2 ghante
python training/train_damage_model.py

# Zyada accuracy (slow)
python training/train_damage_model.py --model yolo11m-seg.pt --epochs 150

# Sirf CPU (bohat slow, sirf test ke liye)
python training/train_damage_model.py --model yolo11n-seg.pt --device cpu --batch 4 --epochs 30
```

Training ke end par test-set ka mAP print hota hai aur best model yahan copy ho jata hai:
`ml_models/car_damage_yolo11-seg.pt` — Django app isay khud load kar leti hai (server restart karein).

Doosri jagah rakha model use karna ho: `DAMAGE_MODEL_PATH=C:\path\best.pt`
Confidence threshold badalna ho: `DAMAGE_MODEL_CONF=0.25` (default 0.35).

## 4. Zyada accuracy chahiye?

1. Pehle bara YOLO lein: `yolo11m-seg` / `yolo11l-seg`, `--imgsz 1024` (patle scratch ke liye madadgar).
2. Apni photos (apni lighting, apne parking area) ka data add karein — ye sab se zyada farq dalta hai.
3. Phir bhi kam ho to Mask R-CNN (Detectron2) try kar sakte hain; `damage_detector.detect()` ka
   output format same rakhein to baqi app badalne ki zarurat nahi.
