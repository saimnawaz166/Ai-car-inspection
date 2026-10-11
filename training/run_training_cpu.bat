@echo off
REM WARNING: laptop par CPU 100%% par ghanton chalta hai. Is laptop par isi wajah se
REM 3 dafa "critical thermal" shutdown hua. Behtar: training\colab_train.ipynb (Google Colab GPU).
REM CPU par YOLO11n-seg training (GPU nahi hai to). Log: runs\train_log.txt
REM Ruk jaye to dobara chalayein: agar runs\car_damage\weights\last.pt mojood hai
REM to training wahin se aage chalegi (resume).
cd /d "%~dp0.."
if exist "runs\car_damage\weights\last.pt" (
    python -u training\train_damage_model.py --resume >> runs\train_log.txt 2>&1
) else (
    python -u training\train_damage_model.py --model yolo11n-seg.pt --imgsz 512 --batch 8 --epochs 30 --patience 10 --device cpu --workers 2 > runs\train_log.txt 2>&1
)
