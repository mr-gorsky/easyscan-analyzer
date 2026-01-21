# ================================
# EasyScan SLO Analyzer – DICOM-AWARE CLINICAL VERSION
# Robust optic disc detection using acquisition metadata
# Designed for i-Optics EasyScan (IR / Green / Merged)
# ================================

import streamlit as st
import cv2
import numpy as np
from PIL import Image
from fractions import Fraction
from datetime import datetime
import base64
import pydicom

# ==================================================
# PAGE SETUP
# ==================================================
st.set_page_config(
    page_title="EasyScan SLO Analyzer (Clinical, DICOM-aware)",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ EasyScan SLO Analyzer – Clinical (DICOM-aware)")
st.caption("Robust analysis for i-Optics EasyScan | Uses DICOM metadata when available")

# ==================================================
# IMAGE UTILS
# ==================================================
def load_image(file):
    if file.name.lower().endswith('.dcm'):
        ds = pydicom.dcmread(file)
        img = ds.pixel_array.astype(np.float32)
        img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        return img, ds
    else:
        img = np.array(Image.open(file))
        return img, None


def to_gray(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    return cv2.GaussianBlur(img, (7, 7), 0)

# ==================================================
# DISC DETECTION – DICOM GUIDED
# ==================================================
def detect_optic_disc(img, ds=None):
    g = to_gray(img)
    h, w = g.shape

    # ---- Use DICOM info if present
    expected_side = None
    if ds is not None:
        desc = str(ds.get('SeriesDescription', '')).lower()
        if 'nasal' in desc:
            expected_side = 'nasal'
        if 'central' in desc:
            expected_side = 'central'

    # ---- CLAHE for IR SLO
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    cl = clahe.apply(g)

    # ---- Gradient ring (disc has strong rim)
    grad = cv2.Laplacian(cl, cv2.CV_32F)
    grad = np.abs(grad)
    grad = cv2.normalize(grad, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    _, rim = cv2.threshold(grad, np.percentile(grad, 95), 255, cv2.THRESH_BINARY)
    rim = cv2.morphologyEx(rim, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(rim, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidates = []

    for c in cnts:
        area = cv2.contourArea(c)
        if not (3000 < area < 70000):
            continue

        peri = cv2.arcLength(c, True)
        if peri == 0:
            continue

        circ = 4 * np.pi * area / (peri ** 2)
        if circ < 0.45:
            continue

        (cx, cy), r = cv2.minEnclosingCircle(c)
        center_dist = np.sqrt((cx - w/2)**2 + (cy - h/2)**2)

        # ---- If central image, reject disc
        if expected_side == 'central' and center_dist < 0.2 * min(w, h):
            continue

        candidates.append((c, r, center_dist))

    if not candidates:
        return None

    # ---- Choose most eccentric strong candidate
    disc_cnt, disc_r, _ = max(candidates, key=lambda x: (x[1], x[2]))
    (cx, cy), _ = cv2.minEnclosingCircle(disc_cnt)

    return {
        'center': (int(cx), int(cy)),
        'radius': int(disc_r),
        'contour': disc_cnt
    }

# ==================================================
# CUP DETECTION
# ==================================================
def detect_cup(img, disc):
    g = to_gray(img)
    cx, cy = disc['center']
    r = disc['radius']

    mask = np.zeros_like(g)
    cv2.circle(mask, (cx, cy), r, 255, -1)
    roi = cv2.bitwise_and(g, g, mask=mask)

    thr = np.mean(roi[mask > 0]) + 0.25 * np.std(roi[mask > 0])
    _, cup_bin = cv2.threshold(roi, thr, 255, cv2.THRESH_BINARY)
    cup_bin = cv2.morphologyEx(cup_bin, cv2.MORPH_OPEN, np.ones((7, 7), np.uint8))

    cnts, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    (_, _), cr = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))
    return int(cr)

# ==================================================
# VESSELS (GREEN)
# ==================================================
def segment_vessels(img):
    g = to_gray(img)
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT,
                          cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15)))
    _, v = cv2.threshold(bh, np.percentile(bh, 90), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(v, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))


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
        'ratio': float(a / v),
        'fraction': f"{frac.numerator}/{frac.denominator}"
    }

# ==================================================
# ANALYSIS PIPELINE
# ==================================================
def analyze_image(file):
    img, ds = load_image(file)

    disc = detect_optic_disc(img, ds)
    result = {}

    if disc:
        cup_r = detect_cup(img, disc)
        if cup_r:
            result['cd_ratio'] = cup_r / disc['radius']
            result['disc'] = disc

    return img, result

# ==================================================
# UI
# ==================================================
st.subheader("Upload IR NASAL image (DICOM preferred)")

file = st.file_uploader("IR Nasal (DICOM .dcm or TIFF)", type=['dcm', 'tif', 'tiff', 'png', 'jpg'])

if file:
    img, res = analyze_image(file)

    st.image(img, caption="Input image", use_container_width=True)

    st.markdown('---')
    st.subheader("Results")

    if 'cd_ratio' in res:
        st.metric("C/D ratio", f"{res['cd_ratio']:.2f}")
    else:
        st.warning("Optic disc not confidently detected – C/D not calculated")

    if 'disc' in res:
        vis = cv2.cvtColor(to_gray(img), cv2.COLOR_GRAY2RGB)
        cx, cy = res['disc']['center']
        cv2.circle(vis, (cx, cy), res['disc']['radius'], (0, 255, 0), 2)
        st.image(vis, caption="Detected optic disc", use_container_width=True)

else:
    st.info("Upload an IR nasal image to begin")
