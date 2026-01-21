# ================================
# EasyScan SLO Analyzer – FULL CLINICAL VERSION
# Optimized for i-Optics EasyScan (IR / Green / Merged)
# Central vs Nasal aware, HARD safety gates for C/D
# ================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fractions import Fraction
from datetime import datetime
import base64

# ==================================================
# PAGE SETUP
# ==================================================
st.set_page_config(
    page_title="EasyScan SLO Analyzer (Clinical)",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ EasyScan SLO Analyzer – Clinical Mode")
st.caption("Designed for i-Optics EasyScan | 6-image protocol per eye")

# ==================================================
# BASIC IMAGE UTILS
# ==================================================
def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(img, (7, 7), 0)

# ==================================================
# OPTIC DISC DETECTION (IR NASAL ONLY)
# ==================================================
def detect_optic_disc_ir_nasal(img):
    g = to_gray(img)
    h, w = g.shape

    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(g)

    # bright local blobs
    _, bright = cv2.threshold(cl, np.percentile(cl, 97), 255, cv2.THRESH_BINARY)
    bright = cv2.morphologyEx(bright, cv2.MORPH_CLOSE, np.ones((15, 15), np.uint8))

    cnts, _ = cv2.findContours(bright, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in cnts:
        area = cv2.contourArea(c)
        if not (2000 < area < 50000):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circ = 4 * np.pi * area / (peri ** 2)
        if circ < 0.5:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)

        # MUST be off-center for nasal
        if center_dist < 0.1 * min(w, h):
            continue

        candidates.append((c, r))

    if not candidates:
        return None

    disc_cnt, disc_r = max(candidates, key=lambda x: x[1])
    (cx, cy), _ = cv2.minEnclosingCircle(disc_cnt)

    return {
        "center": (int(cx), int(cy)),
        "radius": int(disc_r),
        "contour": disc_cnt
    }

# ==================================================
# CUP DETECTION (INSIDE DISC)
# ==================================================
def detect_cup_ir(img, disc):
    g = to_gray(img)
    cx, cy = disc["center"]
    r = disc["radius"]

    mask = np.zeros_like(g)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(g, g, mask=mask)

    thr = np.mean(roi[mask > 0]) + 0.3 * np.std(roi[mask > 0])
    _, cup_bin = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
    cup_bin = cv2.morphologyEx(cup_bin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    (_, _), cr = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return int(cr)

# ==================================================
# MACULA (CENTRAL ONLY)
# ==================================================
def detect_macula_ir_central(img):
    g = to_gray(img)
    h, w = g.shape
    cx, cy = w // 2, h // 2
    roi = g[cy-100:cy+100, cx-100:cx+100]
    return {
        "center": (cx, cy),
        "foveal_reflex": float(np.mean(roi))
    }

# ==================================================
# VESSEL SEGMENTATION (GREEN)
# ==================================================
def segment_vessels_green(img):
    g = to_gray(img)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    _, v = cv2.threshold(bh, np.percentile(bh, 90), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

# ==================================================
# A/V RATIO
# ==================================================
def compute_av_ratio(vessel_bin):
    dist = cv2.distanceTransform(vessel_bin, cv2.DIST_L2, 5)
    widths = dist[vessel_bin > 0] * 2

    if len(widths) < 150:
        return None

    med = np.median(widths)
    a = np.mean(widths[widths < med])
    v = np.mean(widths[widths >= med])

    frac = Fraction(int(a), int(v)).limit_denominator()
    return {
        "ratio": float(a / v),
        "fraction": f"{frac.numerator}/{frac.denominator}"
    }

# ==================================================
# ANALYZE ONE EYE
# ==================================================
def analyze_eye(images):
    results = {}

    # ---- NASAL IR → DISC + CUP
    if "IR_NASAL" in images:
        disc = detect_optic_disc_ir_nasal(images["IR_NASAL"])
        if disc:
            cup_r = detect_cup_ir(images["IR_NASAL"], disc)
            if cup_r:
                results["cd_ratio"] = cup_r / disc["radius"]
                results["disc"] = disc

    # ---- CENTRAL IR → MACULA
    if "IR_CENTRAL" in images:
        results["macula"] = detect_macula_ir_central(images["IR_CENTRAL"])

    # ---- GREEN NASAL → VESSELS + AV
    if "GREEN_NASAL" in images:
        v = segment_vessels_green(images["GREEN_NASAL"])
        results["av"] = compute_av_ratio(v)
        results["vessel_map"] = v

    return results

# ==================================================
# UPLOAD UI (6 IMAGES)
# ==================================================
st.subheader("Upload images – ONE EYE (OD or OS)")

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

# ==================================================
# RUN ANALYSIS
# ==================================================
if st.button("🔬 Analyze Eye"):
    res = analyze_eye(images)

    st.markdown("---")
    st.subheader("Results")

    if "cd_ratio" in res:
        st.metric("C/D ratio (IR nasal)", f"{res['cd_ratio']:.2f}")
    else:
        st.info("Optic disc not confidently detected on IR nasal – C/D not calculated")

    if "av" in res and res["av"]:
        st.metric("A/V ratio", f"{res['av']['ratio']:.2f}")
        st.caption(res['av']['fraction'])

    if "macula" in res:
        st.metric("Foveal reflex (IR central)", f"{res['macula']['foveal_reflex']:.1f}")

    # ---- VISUAL DEBUG OVERLAY
    if "disc" in res and "IR_NASAL" in images:
        vis = cv2.cvtColor(to_gray(images["IR_NASAL"]), cv2.COLOR_GRAY2RGB)
        cx, cy = res["disc"]["center"]
        cv2.circle(vis, (cx, cy), res["disc"]["radius"], (0, 255, 0), 2)
        st.image(vis, caption="IR Nasal – Optic Disc Detected", use_container_width=True)

else:
    st.info("Upload EasyScan images to begin analysis")
