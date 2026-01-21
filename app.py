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
    page_title="EasyScan SLO Analyzer – Advanced",
    page_icon="👁️",
    layout="wide"
)

st.title("👁️ EasyScan Advanced SLO Analyzer")
st.caption("Clinical-grade structural analysis for i-Optics EasyScan SLO")

# ======================================================
# PREPROCESSING (SLO IR)
# ======================================================
def preprocess(img):
    if img.ndim == 3:
        img = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX)
    img = cv2.GaussianBlur(img, (7, 7), 0)
    return img

# ======================================================
# DISC + CUP
# ======================================================
def detect_disc_cup(img):
    g = preprocess(img)

    _, disc_bin = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    disc_bin = cv2.morphologyEx(disc_bin, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))

    cnts, _ = cv2.findContours(disc_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None

    disc_cnt = max(cnts, key=cv2.contourArea)
    (cx, cy), disc_r = cv2.minEnclosingCircle(disc_cnt)

    mask = np.zeros_like(g)
    cv2.circle(mask, (int(cx), int(cy)), int(disc_r), 255, -1)
    roi = cv2.bitwise_and(g, g, mask=mask)

    cup_thr = np.mean(roi[mask > 0]) + 0.3 * np.std(roi[mask > 0])
    _, cup_bin = cv2.threshold(roi, cup_thr, 255, cv2.THRESH_BINARY)
    cup_bin = cv2.morphologyEx(cup_bin, cv2.MORPH_OPEN, np.ones((9, 9), np.uint8))

    cnts, _ = cv2.findContours(cup_bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cup_r = disc_r * 0.4
    if cnts:
        (_, _), cup_r = cv2.minEnclosingCircle(max(cnts, key=cv2.contourArea))

    cd = float(cup_r / disc_r)

    ellipse = cv2.fitEllipse(disc_cnt)
    axis_ratio = max(ellipse[1]) / min(ellipse[1])

    return {
        "center": (int(cx), int(cy)),
        "disc_radius": int(disc_r),
        "cup_radius": int(cup_r),
        "cd_ratio": cd,
        "axis_ratio": axis_ratio
    }

# ======================================================
# ISNT RULE
# ======================================================
def isnt_rule(vessel_map, center):
    cx, cy = center
    h, w = vessel_map.shape

    regions = {
        "I": vessel_map[cy:h, cx-50:cx+50],
        "S": vessel_map[0:cy, cx-50:cx+50],
        "N": vessel_map[cy-50:cy+50, 0:cx],
        "T": vessel_map[cy-50:cy+50, cx:w],
    }

    density = {k: np.sum(v > 0) for k, v in regions.items()}
    return density, density["I"] >= density["S"] >= density["N"] >= density["T"]

# ======================================================
# VESSELS + AV
# ======================================================
def segment_vessels(img):
    g = preprocess(img)
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bh = cv2.morphologyEx(g, cv2.MORPH_BLACKHAT, kernel)
    _, vbin = cv2.threshold(bh, np.percentile(bh, 90), 255, cv2.THRESH_BINARY)
    return cv2.morphologyEx(vbin, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))

def av_ratio(vbin):
    dist = cv2.distanceTransform(vbin, cv2.DIST_L2, 5)
    widths = dist[vbin > 0] * 2
    if len(widths) < 50:
        return None

    med = np.median(widths)
    a = np.mean(widths[widths < med])
    v = np.mean(widths[widths >= med])
    frac = Fraction(int(a), int(v)).limit_denominator()

    return {
        "av": float(a / v),
        "fraction": f"{frac.numerator}/{frac.denominator}"
    }

# ======================================================
# PPA + LESIONS
# ======================================================
def detect_ppa(g, disc):
    cx, cy = disc["center"]
    r = disc["disc_radius"]
    ring = g[cy-r*2:cy+r*2, cx-r*2:cx+r*2]
    return np.std(ring) > 25

def detect_lesions(g):
    _, dark = cv2.threshold(g, 40, 255, cv2.THRESH_BINARY_INV)
    _, bright = cv2.threshold(g, 200, 255, cv2.THRESH_BINARY)
    return np.sum(dark > 0), np.sum(bright > 0)

# ======================================================
# MAIN ANALYSIS
# ======================================================
def analyze(img):
    g = preprocess(img)
    disc = detect_disc_cup(img)
    vessels = segment_vessels(img)
    av = av_ratio(vessels)

    findings = []
    glaucoma_score = 0

    if disc:
        if disc["cd_ratio"] > 0.6:
            findings.append("Enlarged C/D ratio")
            glaucoma_score += 40
        if disc["axis_ratio"] > 1.3:
            findings.append("Tilted optic disc")
            glaucoma_score += 10
        if detect_ppa(g, disc):
            findings.append("Peripapillary atrophy")
            glaucoma_score += 15

        density, isnt_ok = isnt_rule(vessels, disc["center"])
        if not isnt_ok:
            findings.append("ISNT rule violation")
            glaucoma_score += 20

    if av:
        if av["av"] < 0.6:
            findings.append("Arteriolar narrowing")
            glaucoma_score += 10
        elif av["av"] > 0.8:
            findings.append("Venous dilation")

    dark, bright = detect_lesions(g)
    if dark > 500:
        findings.append("Hemorrhage candidates")
    if bright > 500:
        findings.append("Hard exudate candidates")

    glaucoma_score = min(100, glaucoma_score)

    vis = cv2.cvtColor(g, cv2.COLOR_GRAY2RGB)
    if disc:
        cx, cy = disc["center"]
        cv2.circle(vis, (cx, cy), disc["disc_radius"], (0, 255, 0), 2)
        cv2.circle(vis, (cx, cy), disc["cup_radius"], (255, 0, 0), 2)
    vis[vessels > 0] = [255, 165, 0]

    return {
        "disc": disc,
        "av": av,
        "findings": findings,
        "glaucoma_score": glaucoma_score,
        "visualization": vis
    }

# ======================================================
# HTML REPORT
# ======================================================
def generate_report(res):
    html = f"""
    <h1>EasyScan SLO Report</h1>
    <p>Date: {datetime.now()}</p>
    <h2>Optic Disc</h2>
    <p>C/D ratio: {res['disc']['cd_ratio']:.2f}</p>
    <h2>Vessels</h2>
    <p>A/V ratio: {res['av']['av']:.2f} ({res['av']['fraction']})</p>
    <h2>Glaucoma Risk Score</h2>
    <h1>{res['glaucoma_score']}/100</h1>
    <h2>Findings</h2>
    """ + "".join(f"<li>{f}</li>" for f in res["findings"])
    return html

# ======================================================
# UI
# ======================================================
uploaded = st.file_uploader("Upload i-Optics EasyScan SLO image", type=["tif", "tiff", "png", "jpg"])

if uploaded:
    img = np.array(Image.open(uploaded))
    st.image(img, caption="Original SLO", use_container_width=True)

    if st.button("🔬 Run Advanced Analysis"):
        with st.spinner("Analyzing..."):
            res = analyze(img)

        st.image(res["visualization"], caption="Analysis overlay", use_container_width=True)

        st.metric("C/D ratio", f"{res['disc']['cd_ratio']:.2f}")
        st.metric("Glaucoma Risk", f"{res['glaucoma_score']}/100")

        if res["av"]:
            st.metric("A/V ratio", f"{res['av']['av']:.2f}")
            st.caption(res["av"]["fraction"])

        st.subheader("Findings")
        for f in res["findings"]:
            st.warning(f)

        if st.button("📄 Download HTML report"):
            html = generate_report(res)
            b64 = base64.b64encode(html.encode()).decode()
            st.markdown(f"<a href='data:text/html;base64,{b64}' download='slo_report.html'>Download</a>", unsafe_allow_html=True)

else:
    st.info("Upload an EasyScan SLO image to start.")
