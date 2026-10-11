# AutoInspect AI — AI Vehicle Damage Inspection (Django)

Rental car ki **Before** aur **After** photos upload karein → AI har panel compare karta hai →
**Vehicle Inspection Report** (web page + PDF) ban jati hai.

```
Front Bumper — Possible new scratch / minor change
Left Door    — No significant change
Rear Bumper  — Possible damage
```

> Ye ek **AI-assisted inspection tool** hai. Ye legal/insurance-grade assessment nahi hai —
> report mein yeh disclaimer khud likha aata hai.

---

## 1. Setup (pehli dafa)

```bash
cd car_inspection

# virtual environment (recommended)
python -m venv venv
venv\Scripts\activate          # Windows
# source venv/bin/activate     # Mac / Linux

pip install -r requirements.txt
python manage.py migrate
python manage.py createsuperuser   # optional, /admin ke liye
python manage.py runserver
```

Browser mein kholein: **http://127.0.0.1:8000/**

## 2. Demo kaise karein

1. `sample_images/` folder mein 3 jori (pair) photos pehle se hain:
   - `front_bumper_before/after.jpg` → after mein naya **scratch**
   - `rear_bumper_before/after.jpg` → after mein **dent / damage**
   - `left_door_before/after.jpg` → koi damage nahi (sirf angle + roshni alag)
2. **+ New inspection** → car details bharein → har zone mein before/after photo lagayein.
3. **Run AI inspection** dabayein → report khul jayegi → **Download PDF**.

Nayi sample photos banane ke liye: `python make_sample_images.py`

## 3. Folder structure

```
car_inspection/
├── manage.py
├── requirements.txt
├── make_sample_images.py      # demo photos generator
├── sample_images/             # ready demo photos
├── config/                    # Django project settings + main urls
├── inspection/                # main app
│   ├── ai_engine.py           # ⭐ AI comparison logic (OpenCV + SSIM + YOLO)
│   ├── damage_detector.py     # ⭐ YOLO11-seg: damage type + mask + box + confidence
│   ├── report_pdf.py          # PDF report (ReportLab)
│   ├── models.py              # Inspection + ZoneCheck tables
│   ├── forms.py               # vehicle form + photo validation
│   ├── views.py               # pages: home, new, report, pdf, delete
│   ├── urls.py, admin.py
│   └── migrations/
├── training/                  # YOLO11-seg training (data.yaml, CarDD converter, train script)
├── ml_models/                 # trained weights: car_damage_yolo11-seg.pt
├── templates/                 # HTML pages
├── static/                    # CSS + JS
└── media/                     # uploaded + AI-marked photos (runtime)
```

## 4. AI kaise kaam karta hai (`inspection/ai_engine.py`)

| Step | Kya hota hai | Kyun |
|------|--------------|------|
| 1. Resize | Dono photos 800px width par | Same size par compare ho sake |
| 2. Align (ORB + Homography) | After photo ko Before ke angle par ghuma/khiska kar fit karta hai | Staff kabhi bhi bilkul same jagah se photo nahi leta |
| 3. Light normalize (CLAHE) | Brightness/contrast barabar | Dhoop/chaon ko damage na samjhe |
| 4. SSIM map | Har hisse ka "kitna milta hai" score | Shape/texture ka farq pakarta hai (scratch, dent) |
| 5. Brightness check | Sirf wahan flag jahan asal farq bhi wazeh ho | Background ka grain/noise ignore |
| 6. Clean + boxes | Chhote dots hata kar red boxes | Saaf result |
| 7. Decide | Changed area % se result | Neeche thresholds |

**Thresholds** (file ke upar change kar sakte hain):

- `MINOR_CHANGE_PERCENT = 0.15` → is se kam = *No significant change*
- `DAMAGE_PERCENT = 1.5` → is se zyada = *Possible damage*, beech mein = *Possible new scratch*
- Agar photos bilkul match na karein → *Unclear – please retake photo*

## 4b. Damage type detection (YOLO11-seg)

Trained model ho to AI har damage ki **class, mask, bounding box aur confidence** bhi deta hai:

| Class | Key |
|-------|-----|
| Scratch | `scratch` |
| Dent | `dent` |
| Crack | `crack` |
| Broken lamp | `broken_lamp` |
| Glass shatter | `glass_shatter` |
| Flat tire | `flat_tire` |

- Model **Before aur After dono** photos par chalta hai. After ka jo damage Before mein
  usi jagah pehle se tha, woh *"Already in before photo"* (gray, "old") mark hota hai —
  result sirf **naye** damage se badalta hai.
- Naya scratch (chhota) = *Possible new scratch*; dent / crack / lamp / glass / tire ya bara
  scratch = *Possible damage*.
- Har damage DB mein `Detection` table mein save hota hai (label, confidence, x1..y2, polygon mask,
  area %, is_new) aur report + PDF mein table ki shakal mein dikhta hai.
- Model file na ho ya `ultralytics` install na ho to app pehle ki tarah sirf SSIM se chalti hai.

Model train karne ka tareeqa: **[training/README.md](training/README.md)**

Command line se ek photo test karna:

```bash
python -m inspection.damage_detector photo.jpg --save marked.jpg
```

## 5. Achhe results ke liye photo tips

- Har panel ki photo **same angle, same distance** se lein (zameen par marker laga dein to best).
- Panel frame ka zyada hissa bharay; background kam ho.
- Seedhi dhoop / tez reflection se bachein.
- Har inspection mein sirf **4 photos**: Front, Back, Left side, Right side — poori gaadi frame
  mein, 2-3 meter door se. AI pehle photo mein gaadi pehchanta hai (background ignore), phir
  har damage ki class + jagah batata hai, e.g. "Scratch · Rear bumper (left)".
- Raat ko flash ke saath vs din ki roshni — ye "unclear" ya ghalat flag de sakta hai.

## 6. Aage kya add kar sakte hain

- Customer ka signature pad + email par PDF bhejna.
- Login (staff/customer roles) aur rental booking se link.
