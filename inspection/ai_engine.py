"""
AI-assisted damage comparison engine.

Kaam: ek hi car panel ki "Before" aur "After" photo ko compare karna
aur batana ke kahin naya scratch / dent / change to nahi aaya.

Steps (simple words mein):
  1. Dono photos load karo aur ek size par le aao.
  2. "After" photo ko "Before" ke upar ALIGN karo (kyunke dono photos
     bilkul same angle se nahi li jatin). Iske liye ORB feature matching
     + homography use hota hai.
  3. Lighting ka farq kam karo (CLAHE) taa-ke dhoop/chaon ko damage na samjha jaye.
  4. SSIM (Structural Similarity) se pixel-level farq ka "map" banao.
  5. Jahan farq zyada hai un hisson ko boxes mein mark karo.
  6. Farq kitna hai (% area) uske hisaab se result do:
        No significant change / Possible new scratch / Possible damage
  7. (Agar trained model mojood ho) YOLO11-seg dono photos par chalao,
     damage ki class (scratch, dent, crack, broken lamp, glass shatter,
     flat tire) + mask + box + confidence nikalo. Jo damage Before mein
     pehle se tha usay "old" mark karo; sirf NAYA damage result badalta hai.
     Dekhein: damage_detector.py

NOTE: Ye ek AI-assisted tool hai, insurance/legal grade inspection nahi.
"""

import cv2
import numpy as np
from skimage.metrics import structural_similarity

from . import damage_detector

# --- Tunable settings (inko change kar ke sensitivity adjust kar sakte hain) ---
WORK_WIDTH = 1280            # Comparison se pehle photos is width par resize hongi (poori gaadi ki photo)
LOCAL_DIFF_THRESHOLD = 0.45  # SSIM map mein is se zyada farq = "changed pixel"
PIXEL_DIFF_THRESHOLD = 30    # Brightness ka kam az kam itna farq (0-255) hona chahiye
MIN_REGION_FRACTION = 0.001  # Is se chhote patches ignore (noise)
MINOR_CHANGE_PERCENT = 0.15  # Is se kam = No significant change
DAMAGE_PERCENT = 1.5         # Is se zyada = Possible damage
UNCLEAR_SSIM = 0.25          # Agar overall similarity is se kam ho to photos match hi nahi kar rahi
EXISTING_OVERLAP = 0.4       # After ke damage mask ka itna hissa Before ke same-class damage par ho = purana damage
SCRATCH_DAMAGE_PERCENT = 2.0 # Itna bara scratch (photo ka %) = "Possible damage", chhota = "minor"

STATUS_NO_CHANGE = "no_change"
STATUS_MINOR = "minor"
STATUS_DAMAGE = "damage"
STATUS_UNCLEAR = "unclear"

STATUS_LABELS = {
    STATUS_NO_CHANGE: "No significant change",
    STATUS_MINOR: "Possible new scratch / minor change",
    STATUS_DAMAGE: "Possible damage",
    STATUS_UNCLEAR: "Unclear - please retake photo from the same angle",
}


def _load_and_resize(path, width=WORK_WIDTH):
    """Photo ko disk se parho aur fixed width par resize karo."""
    img = cv2.imread(str(path))
    if img is None:
        raise ValueError(f"Image read nahi ho saki: {path}")
    h, w = img.shape[:2]
    scale = width / float(w)
    return cv2.resize(img, (width, int(h * scale)), interpolation=cv2.INTER_AREA)


def _align(before, after, mask_b=None, mask_a=None):
    """
    'After' photo ko 'Before' photo ke angle par set karo.

    ORB dono photos mein khaas points (corners, edges) dhoondta hai,
    phir unko match kar ke ek transformation (homography) nikalte hain
    jo After ko Before ke upar fit kar deti hai.

    mask_b / mask_a: gaadi ke masks — points sirf gaadi par dhoondo, background par nahi
    (warna alignment deewar ya zameen ke hisaab se ho jati hai).

    Returns: (aligned_after, valid_mask, aligned_ok)
    """
    h, w = before.shape[:2]
    full_mask = np.full((h, w), 255, dtype=np.uint8)
    after_resized = cv2.resize(after, (w, h))

    orb = cv2.ORB_create(3000)
    gray_b = cv2.cvtColor(before, cv2.COLOR_BGR2GRAY)
    gray_a = cv2.cvtColor(after_resized, cv2.COLOR_BGR2GRAY)
    m_b = mask_b.astype(np.uint8) * 255 if mask_b is not None else None
    m_a = cv2.resize(mask_a.astype(np.uint8) * 255, (w, h), interpolation=cv2.INTER_NEAREST)         if mask_a is not None else None
    kp_b, des_b = orb.detectAndCompute(gray_b, m_b)
    kp_a, des_a = orb.detectAndCompute(gray_a, m_a)

    if des_b is None or des_a is None or len(kp_b) < 10 or len(kp_a) < 10:
        return after_resized, full_mask, False

    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
    matches = sorted(matcher.match(des_a, des_b), key=lambda m: m.distance)
    good = matches[: max(10, int(len(matches) * 0.2))]
    if len(good) < 10:
        return after_resized, full_mask, False

    src = np.float32([kp_a[m.queryIdx].pt for m in good]).reshape(-1, 1, 2)
    dst = np.float32([kp_b[m.trainIdx].pt for m in good]).reshape(-1, 1, 2)
    H, inliers = cv2.findHomography(src, dst, cv2.RANSAC, 5.0)

    if H is None or inliers is None or inliers.sum() < 10:
        return after_resized, full_mask, False

    # Sanity check: agar transformation bohat ajeeb hai (photo ulat gayi etc.) to use na karo
    det = np.linalg.det(H[:2, :2])
    if not (0.5 < det < 2.0):
        return after_resized, full_mask, False

    aligned = cv2.warpPerspective(after_resized, H, (w, h))
    # Warp ke baad kinaron par kaala (empty) hissa aa jata hai, usay compare se bahar rakho
    valid = cv2.warpPerspective(full_mask, H, (w, h))
    valid = cv2.erode(valid, np.ones((15, 15), np.uint8))
    return aligned, valid, True


def _normalize_light(img):
    """CLAHE se brightness/contrast barabar karo, phir grayscale + blur."""
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    l = clahe.apply(l)
    gray = cv2.cvtColor(cv2.merge([l, a, b]), cv2.COLOR_LAB2BGR)
    gray = cv2.cvtColor(gray, cv2.COLOR_BGR2GRAY)
    return cv2.GaussianBlur(gray, (5, 5), 0)


def _mark_new_damage(before_dets, after_dets, aligned_ok):
    """
    After photo ke har damage par is_new = True/False lagao.

    Alignment kaamyab ho to After aur Before ek hi coordinates mein hain,
    is liye mask overlap se pata chalta hai ke ye damage pehle se tha ya nahi.
    Alignment fail ho to sirf class dekhte hain (Before mein same class thi = purana).
    """
    for d in after_dets:
        same = [b for b in before_dets if b["label"] == d["label"]]
        if not same:
            d["is_new"] = True
        elif not aligned_ok:
            d["is_new"] = False
        else:
            old_mask = np.logical_or.reduce([b["mask"] for b in same])
            area = max(int(d["mask"].sum()), 1)
            overlap = np.logical_and(d["mask"], old_mask).sum() / area
            d["is_new"] = bool(overlap < EXISTING_OVERLAP)


def _damage_status(new_dets):
    """Naye damage ki class se status."""
    if not new_dets:
        return STATUS_NO_CHANGE
    for d in new_dets:
        if d["label"] != "scratch" or d["area_percent"] >= SCRATCH_DAMAGE_PERCENT:
            return STATUS_DAMAGE
    return STATUS_MINOR


def compare_images(before_path, after_path, annotated_out_path=None, view=None):
    """
    Main function. Before aur After photo ka path do, result dictionary milegi.

    view: 'front' | 'back' | 'left' | 'right' — poori gaadi ki photo kis taraf se hai.
          Isse har damage ki jagah (part) nikalti hai, e.g. "Headlight (right)".

    Returns dict:
        status          -> 'no_change' | 'minor' | 'damage' | 'unclear'
        label           -> insaan ke parhne wala result
        change_percent  -> panel ka kitna % hissa badla hua laga
        similarity      -> 0-100, dono photos kitni milti julti hain
        regions         -> kitne alag jagah changes mile
        aligned         -> kya auto-alignment kaamyab hua
        annotated_path  -> red boxes wali image ka path (agar save ki)
        detector        -> AI model ka naam, ya None agar model available nahi
        detections      -> After photo ke damages (YOLO11-seg), har ek:
                           label, confidence, bbox [x1,y1,x2,y2], polygon [[x,y],..],
                           mask (bool array), area_percent, is_new, part
                           (pixels annotated image ke coordinates mein hain)
        car_found       -> kya photo mein gaadi pehchani gayi (comparison sirf usi par hua)
    """
    before = _load_and_resize(before_path)
    after = _load_and_resize(after_path)

    # Gaadi ka outline: comparison + detection sirf gaadi par, background (log, deewar) par nahi
    car_b = damage_detector.car_mask(before)
    car_a = damage_detector.car_mask(after)

    aligned_after, valid_mask, aligned_ok = _align(before, after, car_b, car_a)
    h, w = before.shape[:2]
    frame = (0, 0, w, h)
    if car_b is not None:
        car_region = cv2.dilate(car_b.astype(np.uint8) * 255, np.ones((15, 15), np.uint8))
        valid_mask = cv2.bitwise_and(valid_mask, car_region)
        ys, xs = np.nonzero(car_b)
        frame = (int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max()))

    g_before = _normalize_light(before)
    g_after = _normalize_light(aligned_after)

    # SSIM: 1.0 = bilkul same, 0 = bilkul alag. full=True se har pixel ka score milta hai
    score, ssim_map = structural_similarity(g_before, g_after, full=True, data_range=255)
    diff = 1.0 - ssim_map  # ab zyada value = zyada farq
    if car_b is not None and np.count_nonzero(valid_mask):
        score = float(ssim_map[valid_mask > 0].mean())  # similarity sirf gaadi par

    # Doosra check: asal brightness ka farq bhi kaafi hona chahiye.
    # Isse background ka grain/noise (jo SSIM ko confuse karta hai) ignore ho jata hai,
    # jab ke scratch/dent ka farq wazeh hota hai.
    smooth_b = cv2.GaussianBlur(g_before, (7, 7), 0).astype(np.int16)
    smooth_a = cv2.GaussianBlur(g_after, (7, 7), 0).astype(np.int16)
    smooth_a = smooth_a - int(np.median(smooth_a - smooth_b))  # overall roshni ka farq nikaal do
    abs_diff = np.abs(smooth_a - smooth_b)

    changed = ((diff > LOCAL_DIFF_THRESHOLD) & (abs_diff > PIXEL_DIFF_THRESHOLD)).astype(np.uint8) * 255
    changed = cv2.bitwise_and(changed, valid_mask)  # alignment ke kaale kinare hata do

    # Morphology: chhote dots (noise) saaf karo, paas paas ke hisse jor do
    changed = cv2.morphologyEx(changed, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
    changed = cv2.morphologyEx(changed, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    valid_area = max(int(np.count_nonzero(valid_mask)), 1)
    min_area = MIN_REGION_FRACTION * valid_area

    contours, _ = cv2.findContours(changed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    regions = [c for c in contours if cv2.contourArea(c) >= min_area]

    region_mask = np.zeros_like(changed)
    cv2.drawContours(region_mask, regions, -1, 255, thickness=cv2.FILLED)
    changed_pixels = int(np.count_nonzero(cv2.bitwise_and(changed, region_mask)))
    change_percent = round(100.0 * changed_pixels / valid_area, 2)

    # ---- Result decide karo ----
    if score < UNCLEAR_SSIM:
        status = STATUS_UNCLEAR
    elif change_percent < MINOR_CHANGE_PERCENT or not regions:
        status = STATUS_NO_CHANGE
    elif change_percent < DAMAGE_PERCENT:
        status = STATUS_MINOR
    else:
        status = STATUS_DAMAGE

    # ---- Trained model se damage type (agar available ho) ----
    detections, detector = [], None
    if damage_detector.is_available():
        detector = damage_detector.status_message()
        # Model mojood ho to "Possible damage" sirf asal pehchana hua damage de sakta hai.
        # Akela comparison (dhoop, reflection, gard se bhi badalta hai) zyada se zyada "minor".
        if status == STATUS_DAMAGE:
            status = STATUS_MINOR
        before_dets = damage_detector.detect(before)
        detections = damage_detector.detect(aligned_after)
        # Alignment ke kaale kinaron par aayi detections hata do
        detections = [d for d in detections
                      if np.logical_and(d["mask"], valid_mask > 0).sum() > 0.5 * d["mask"].sum()]
        _mark_new_damage(before_dets, detections, aligned_ok)
        for d in detections:
            d["part"] = damage_detector.locate(d["label"], view, d["bbox"], frame) if view else ""
        ai_status = _damage_status([d for d in detections if d["is_new"]])
        status = overall_status([status, ai_status])

    annotated_path = None
    if annotated_out_path:
        annotated = aligned_after.copy()
        # Halki red overlay changed hisse par
        overlay = annotated.copy()
        overlay[region_mask > 0] = (0, 0, 255)
        annotated = cv2.addWeighted(overlay, 0.25, annotated, 0.75, 0)
        for c in regions:
            x, y, w, h = cv2.boundingRect(c)
            pad = 6
            cv2.rectangle(annotated, (x - pad, y - pad), (x + w + pad, y + h + pad), (0, 0, 255), 2)
        if detections:
            annotated = damage_detector.draw_detections(annotated, detections)
        cv2.imwrite(str(annotated_out_path), annotated)
        annotated_path = str(annotated_out_path)

    return {
        "status": status,
        "label": STATUS_LABELS[status],
        "change_percent": change_percent,
        "similarity": round(float(score) * 100, 1),
        "regions": len(regions),
        "aligned": aligned_ok,
        "annotated_path": annotated_path,
        "detector": detector,
        "detections": detections,
        "car_found": car_b is not None,
    }


def overall_status(statuses):
    """Saare panels ke results se poori car ka ek overall result."""
    if STATUS_DAMAGE in statuses:
        return STATUS_DAMAGE
    if STATUS_MINOR in statuses:
        return STATUS_MINOR
    if STATUS_UNCLEAR in statuses:
        return STATUS_UNCLEAR
    return STATUS_NO_CHANGE
