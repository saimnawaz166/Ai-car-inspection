"""
Demo ke liye sample Before/After photos banata hai (asli car photos ke bajaye).

Chalane ka tareeqa:   python make_sample_images.py
Photos 'sample_images/' folder mein ban jayengi.

- front_bumper_after.jpg  -> naya scratch daala gaya hai
- rear_bumper_after.jpg   -> dent / bara damage daala gaya hai
- left_door_after.jpg     -> koi damage nahi (sirf angle aur roshni thori alag)
"""
from pathlib import Path

import cv2
import numpy as np

OUT = Path(__file__).parent / "sample_images"
OUT.mkdir(exist_ok=True)
rng = np.random.default_rng(7)


def make_panel(color, seed):
    """Ek fake car panel: background, body color gradient, panel lines, handle, logo."""
    r = np.random.default_rng(seed)
    h, w = 600, 900
    img = np.zeros((h, w, 3), np.uint8)
    # parking floor / background texture
    img[:] = (90, 95, 100)
    noise = r.integers(0, 40, (h, w, 1), dtype=np.uint8)
    img = cv2.add(img, np.repeat(noise, 3, axis=2))
    # car body with vertical gradient
    body = np.zeros((420, 820, 3), np.float32)
    for i in range(420):
        body[i] = np.array(color, np.float32) * (0.75 + 0.45 * (i / 420))
    img[90:510, 40:860] = np.clip(body, 0, 255).astype(np.uint8)
    # panel lines, details (yeh features alignment mein madad karte hain)
    cv2.line(img, (40, 200), (860, 210), (30, 30, 30), 3)
    cv2.line(img, (450, 90), (455, 510), (30, 30, 30), 2)
    cv2.rectangle(img, (120, 260), (260, 290), (200, 200, 200), -1)
    cv2.rectangle(img, (560, 265), (700, 295), (200, 200, 200), -1)
    cv2.circle(img, (450, 380), 35, (210, 210, 210), 4)
    cv2.putText(img, "GT", (420, 395), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (220, 220, 220), 3)
    for x in range(80, 840, 60):
        cv2.circle(img, (x, 470), 6, (40, 40, 40), -1)
    # reflection
    cv2.ellipse(img, (300, 150), (160, 25), -5, 0, 360, (240, 240, 240), -1)
    return cv2.GaussianBlur(img, (3, 3), 0)


def shoot_again(img, dx=12, dy=-8, angle=1.5, brightness=18):
    """Same panel ko thore se alag angle aur roshni mein dobara 'photo' karna."""
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), angle, 1.02)
    M[:, 2] += (dx, dy)
    out = cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)
    out = cv2.convertScaleAbs(out, alpha=1.0, beta=brightness)
    noise = rng.normal(0, 4, out.shape)
    return np.clip(out.astype(np.float32) + noise, 0, 255).astype(np.uint8)


def add_scratch(img):
    out = img.copy()
    pts = [(560, 330), (640, 352), (720, 372), (790, 395)]
    for i in range(len(pts) - 1):
        cv2.line(out, pts[i], pts[i + 1], (235, 235, 235), 3)
        cv2.line(out, (pts[i][0], pts[i][1] + 8), (pts[i + 1][0], pts[i + 1][1] + 8), (25, 25, 25), 2)
    return out


def add_dent(img):
    out = img.copy()
    cv2.ellipse(out, (260, 400), (120, 70), 15, 0, 360, (25, 25, 30), -1)
    cv2.ellipse(out, (240, 385), (70, 30), 15, 0, 360, (190, 190, 200), -1)
    cv2.line(out, (150, 330), (380, 470), (240, 240, 240), 4)
    return out


panels = {
    "front_bumper": ((40, 40, 170), add_scratch),   # red car (BGR)
    "rear_bumper": ((40, 40, 170), add_dent),
    "left_door": ((40, 40, 170), None),
}

for i, (name, (color, damage_fn)) in enumerate(panels.items()):
    before = make_panel(color, seed=i)
    after = damage_fn(before) if damage_fn else before
    after = shoot_again(after)
    cv2.imwrite(str(OUT / f"{name}_before.jpg"), before)
    cv2.imwrite(str(OUT / f"{name}_after.jpg"), after)
    print("saved", name)
