import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Stego‑Interference Detector")

# ─── Fixed Settings ────────────────────────────────────────────────────
P_EXPONENT     = 2.5        # block_dgc exponent
WEIGHT_EXP     = 2          # block weight = sum(mag**WEIGHT_EXP)
GRAD_THRESHOLD = 1.0        # ignore low‑energy blocks
BLOCK_SIZE     = 7          # block size for DGC

# ─── Empirical Medians ─────────────────────────────────────────────────
MEDIAN_CLEAN   = 0.0030     # clean images median DGC
MEDIAN_STEGO   = 0.0018     # stego images median DGC

LOW_MEDIAN  = min(MEDIAN_CLEAN, MEDIAN_STEGO)
HIGH_MEDIAN = max(MEDIAN_CLEAN, MEDIAN_STEGO)

# ─── Helpers ──────────────────────────────────────────────────────────
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ori = np.arctan2(Gy, Gx) % (2*pi)
    return mag, ori

def block_dgc(mb, ob):
    w = np.sum(mb) + 1e-8
    sum_cos = np.sum(mb * np.cos(ob))
    sum_sin = np.sum(mb * np.sin(ob))
    R = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (R / w)
    return sigma ** P_EXPONENT

def compute_weighted_dgc_score(img):
    mag, ori = compute_gradients(img)
    h, w = img.shape
    num = den = 0.0
    for i in range(0, h - BLOCK_SIZE + 1, BLOCK_SIZE):
        for j in range(0, w - BLOCK_SIZE + 1, BLOCK_SIZE):
            mb = mag[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            ob = ori[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            weight = np.sum(mb ** WEIGHT_EXP)
            if weight < GRAD_THRESHOLD:
                continue
            num += block_dgc(mb, ob) * weight
            den += weight
    return (num / den) if den > 0 else 0.0

def get_wavelet_detail_image(gray):
    _, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

def get_wavelet_denoised_image(detail_img):
    return cv2.bilateralFilter(detail_img, d=9, sigmaColor=75, sigmaSpace=75)

# ─── App Body ─────────────────────────────────────────────────────────
uploaded = st.file_uploader("Upload a Grayscale Image", type=['png','jpg','jpeg'])
if not uploaded:
    st.stop()

# Load and decode
data = np.frombuffer(uploaded.read(), np.uint8)
gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
if gray is None:
    st.error("Invalid image.")
    st.stop()

# Compute detail and denoised images
detail   = get_wavelet_detail_image(gray)
denoised = get_wavelet_denoised_image(detail)

# Display original & denoised side by side
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Grayscale")
    st.image(gray, clamp=True, channels="GRAY", use_container_width=True)
with col2:
    st.subheader("Denoised Detail")
    st.image(denoised, clamp=True, channels="GRAY", use_container_width=True)

# ─── compute fused score as before ───────────────────────────────────
raw_score      = compute_weighted_dgc_score(detail)
denoised_score = compute_weighted_dgc_score(denoised)
fused          = raw_score - denoised_score

# ─── relative position (no clamp) ───────────────────────────────────
rel_pos = (fused - LOW_MEDIAN) / (HIGH_MEDIAN - LOW_MEDIAN) * 100

# ─── draw number line from 0…100 with marker at rel_pos ─────────────
fig2, ax2 = plt.subplots(figsize=(8,1.5))
ax2.hlines(0, 0, 100, colors='gray', linewidth=4)
ax2.plot(rel_pos, 0, 'o', markersize=10, color='red')
ax2.text(0, 0.1, 'Clean median', ha='left', va='bottom')
ax2.text(100, 0.1, 'Stego median', ha='right', va='bottom')
ax2.text(rel_pos, -0.1, f"{rel_pos:.1f}%", ha='center', va='top', color='red')
ax2.axis('off')
ax2.set_xlim(-20, 120)
ax2.set_ylim(-0.5, 0.5)
st.pyplot(fig2)

# Display fused and raw/denoised scores
st.markdown(f"**Raw DGC Score:**      {raw_score:.4f}")
st.markdown(f"**Denoised DGC Score:** {denoised_score:.4f}")
st.markdown(f"**Difference:**        {fused:.4f}")

# Difference map visualization
diff_map = cv2.absdiff(detail, denoised)
st.subheader("Detail − Denoised Difference")
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(diff_map, cmap='inferno')
ax.axis('off')
st.pyplot(fig)

# Explanation
st.markdown("""
**Difference Map Explained:**  
Bright spots indicate where smoothing removed the most high‑frequency bumps—these are likely embedding sites.  

**Relative Position:**  
The red marker shows where your image falls between the clean median (0%) and stego median (100%) of fused‑difference scores.
""")
