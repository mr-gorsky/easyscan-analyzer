import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fractions import Fraction
from datetime import datetime
import base64

# ======================================================
# PAGE SETUP
# ======================================================
st.set_page_config(
    page_title="EasyScan SLO Analyzer (Clinical)",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ EasyScan SLO Analyzer – Clinical Mode")
st.caption("Structured analysis for i-Optics EasyScan (6-image protocol per eye)")

# ======================================================
# UTILITIES
# ======================================================
def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(img, (7, 7), 0)

# ======================================================
# OPTIC DISC (NASAL ONLY)
# ======================================================
def detect_optic_disc(img):
    g = to_gray(img)

    _, bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    bin = cv2.morphologyEx(bin, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    cnts, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    valid = []

    for c in cnts:
        area = cv2.contourArea(c)
        if not (4000 < area < 60000):
            continue
        peri = cv2.arcLength(c, True)
        circ = 4 * np.pi * area / (peri ** 2)
        if circ < 0.6:
            continue
        valid.append(c)

    if not valid:
        return None

    disc = max(valid, key=cv2.contourArea)
    (cx, cy), r = cv2.minEnclosingCircle(disc)

    h, w = g.shape
    if abs(cx - w/2) < w*0.08 and abs(cy - h/2) < h*0.08:
        return None  # central → not disc

    return {"center": (int(cx), int(cy)), "radius": int(r), "contour": disc}

# ======================================================
# CUP
# ======================================================
def detect_cup(g, disc):
    cx, cy = disc["center"]
    r = disc["radius"]

    mask = np.zeros_like(g)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(g, g, mask=mask)

    thr = np.mean(roi[mask > 0]) + 0.3*np.std(roi[mask > 0])
    _, cup_bin = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
    cup_bin = cv2.morphologyEx(cup_bin, cv2.MORPH_OPEN, np.ones((9,9), np.uint8))

    cnts, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    (_, _), cr = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return int(cr)

# ======================================================
# MACULA (CENTRAL ONLY)
# ======================================================
def detect_macula(img):
    g = to_gray(img)
    h, w = g.shape
    cx, cy = w//2, h//2

    roi = g[cy-100:cy+100, cx-100:cx+100]
    reflex = np.mean(roi)

    return {
        "center": (cx, cy),
        "foveal_reflex": reflex
    }

# ======================================================
# VESSELS
# ======================================================
def segment_vessels(img):
    g = to_gray(img)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15,15)))
    _, v = cv2.threshold(bh, np.percentile(bh, 90), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3,3), np.uint8))

def av_ratio(v):
    dist = cv2.distanceTransform(v, cv2.DIST_L2, 5)
    w = dist[v > 0] * 2
    if len(w) < 100:
        return None
    med = np.median(w)
    a, ve = np.mean(w[w < med]), np.mean(w[w >= med])
    frac = Fraction(int(a), int(ve)).limit_denominator()
    return {"ratio": a/ve, "fraction": f"{frac.numerator}/{frac.denominator}"}

# ======================================================
# ANALYSIS PIPELINE
# ======================================================
def analyze_eye(images):
    results = {}

    # NASAL IR → DISC
    if "IR_NASAL" in images:
        g = to_gray(images["IR_NASAL"])
        disc = detect_optic_disc(images["IR_NASAL"])
        if disc:
            cup_r = detect_cup(g, disc)
            if cup_r:
                cd = cup_r / disc["radius"]
                results["C/D"] = cd
                results["disc"] = disc

    # CENTRAL IR → MACULA
    if "IR_CENTRAL" in images:
        results["macula"] = detect_macula(images["IR_CENTRAL"])

    # VESSELS (GREEN NASAL)
    if "GREEN_NASAL" in images:
        v = segment_vessels(images["GREEN_NASAL"])
        results["AV"] = av_ratio(v)
        results["vessel_map"] = v

    return results

# ======================================================
# UPLOAD UI
# ======================================================
st.subheader("Upload images – one eye")

labels = {
    "IR_CENTRAL": "IR Central",
    "GREEN_CENTRAL": "Green Central",
    "MERGED_CENTRAL": "Merged Central",
    "IR_NASAL": "IR Nasal",
    "GREEN_NASAL": "Green Nasal",
    "MERGED_NASAL": "Merged Nasal",
}

images = {}
cols = st.columns(3)

for i, (k, label) in enumerate(labels.items()):
    with cols[i % 3]:
        f = st.file_uploader(label, type=["tif", "tiff", "png", "jpg"], key=k)
        if f:
            images[k] = np.array(Image.open(f))
            st.image(images[k], use_container_width=True)

# ======================================================
# RUN
# ======================================================
if st.button("🔬 Analyze Eye"):
    res = analyze_eye(images)

    st.markdown("---")
    st.subheader("Results")

    if "C/D" in res:
        st.metric("C/D ratio (nasal)", f"{res['C/D']:.2f}")
    else:
        st.info("Optic disc not in field of view – C/D not calculated")

    if "AV" in res and res["AV"]:
        st.metric("A/V ratio", f"{res['AV']['ratio']:.2f}")
        st.caption(res["AV"]["fraction"])

    if "macula" in res:
        st.metric("Foveal reflex (IR central)", f"{res['macula']['foveal_reflex']:.1f}")

