import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Stego‑Interference Detector")

# ─── Fixed Settings ────────────────────────────────────────────────────
P_EXPONENT     = 2.5        # block_dgc exponent
WEIGHT_EXP     = 2          # block weight = sum(mag**WEIGHT_EXP)
GRAD_THRESHOLD = 1.0        # ignore low‑energy blocks
BLOCK_SIZE     = 7          # block size for DGC

# ─── Empirical Pivot & Steepness ──────────────────────────────────────
PIVOT_T   = 0.00805         # midpoint of clean/stego medians
K_STEEP   = 200             # higher → sharper 50% transition

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
    return (num/den) if den > 0 else 0.0

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

# Read and decode
data = np.frombuffer(uploaded.read(), np.uint8)
gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
if gray is None:
    st.error("Invalid image.")
    st.stop()

# Compute detail + denoised maps
detail   = get_wavelet_detail_image(gray)
denoised = get_wavelet_denoised_image(detail)

# — Show original and denoised side by side —
col1, col2 = st.columns(2)
with col1:
    st.subheader("Original Grayscale")
    st.image(gray, clamp=True, channels="GRAY", use_column_width=True)
with col2:
    st.subheader("Denoised Detail")
    st.image(denoised, clamp=True, channels="GRAY", use_column_width=True)

# Compute DGC scores
raw_score      = compute_weighted_dgc_score(detail)
denoised_score = compute_weighted_dgc_score(denoised)
fused          = raw_score - denoised_score

# Logistic mapping → [0,100]%
likelihood = 1 / (1 + exp(-K_STEEP * (fused - PIVOT_T)))
likelihood *= 100

# Display core metrics
st.markdown(f"**Raw DGC Score:**      {raw_score:.4f}")
st.markdown(f"**Denoised DGC Score:** {denoised_score:.4f}")
st.markdown(f"**Difference:**        {fused:.4f}")
st.markdown(f"### Likelihood of Stego Interference: {likelihood:.1f}%")

# Difference map visualization
diff_map = cv2.absdiff(detail, denoised)
st.subheader("Detail − Denoised Difference")
fig, ax = plt.subplots(figsize=(5,5))
ax.imshow(diff_map, cmap='inferno')
ax.axis('off')
st.pyplot(fig)

# Explanation of the difference map
st.markdown("""
**Difference Map Explained:**  
Bright regions show exactly where smoothing has erased the most hidden “bumps” in the texture.  
Those hotspots correspond to areas where secret data was embedded, and their strength drives the likelihood score above.
""")
