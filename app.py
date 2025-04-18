import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt

st.set_page_config(layout="wide")
st.title("Stego‑Interference Detector with Visualizations")

# ─── Fixed Settings ────────────────────────────────────────────────────
P_EXPONENT     = 2.5        # block_dgc exponent
WEIGHT_EXP     = 2          # block weight = sum(mag**WEIGHT_EXP)
GRAD_THRESHOLD = 1.0        # ignore low‑energy blocks
BLOCK_SIZE     = 7          # block size for DGC
PIVOT_T        = 0.01611    # pivot for 50% likelihood

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

if uploaded:
    # Read images
    data = np.frombuffer(uploaded.read(), np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Invalid image.")
        st.stop()

    # Compute detail & denoised
    detail   = get_wavelet_detail_image(gray)
    denoised = get_wavelet_denoised_image(detail)

    # Compute DGC scores
    raw_score      = compute_weighted_dgc_score(detail)
    denoised_score = compute_weighted_dgc_score(denoised)
    fused          = raw_score - denoised_score

    # Piecewise stretch to %age
    if fused <= PIVOT_T:
        likelihood = 50.0 * (fused / PIVOT_T)
    else:
        likelihood = 50.0 + 50.0 * ((fused - PIVOT_T) / (1.0 - PIVOT_T))

    # Display core metrics
    st.markdown(f"**Raw DGC Score:**      {raw_score:.4f}")
    st.markdown(f"**Denoised DGC Score:** {denoised_score:.4f}")
    st.markdown(f"**Difference:**        {fused:.4f}")
    st.markdown(f"### Likelihood of Stego Interference: {likelihood:.1f}%")

    # ─── ELI5 Explanation ───────────────────────────────────────────────
    st.write("""
    **ELI5:**  
    1. We turn the picture into a bumpy terrain map of edges (“wavelet detail”).  
    2. We smooth it (“denoise”) to wipe out secret bits.  
    3. We measure how “wobbly” each patch is before vs. after: that’s the DGC score.  
    4. The bigger the drop when we smooth, the more likely someone hid data there.  
    5. We turn that drop into a percentage to tell you “how much” looks tampered.
    """)

    # ─── 1) Block‑Level DGC Heatmap ──────────────────────────────────────
    mag, ori = compute_gradients(detail)
    H, W = detail.shape
    heatmap = np.zeros_like(detail, dtype=float)

    # compute block DGC for each patch
    for i in range(0, H - BLOCK_SIZE + 1, BLOCK_SIZE):
        for j in range(0, W - BLOCK_SIZE + 1, BLOCK_SIZE):
            mb = mag[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            ob = ori[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE]
            sigma = block_dgc(mb, ob
            heatmap[i:i+BLOCK_SIZE, j:j+BLOCK_SIZE] = sigma

    # overlay heatmap on detail image
    fig1, ax1 = plt.subplots(figsize=(4,4))
    ax1.imshow(detail, cmap='gray')
    hm = ax1.imshow(heatmap, cmap='jet', alpha=0.5, vmin=0, vmax=1)
    ax1.set_title("Block‑Level DGC Heatmap")
    ax1.axis('off')
    fig1.colorbar(hm, ax=ax1, fraction=0.046, pad=0.04)
    st.pyplot(fig1)

    # ─── 2) Raw vs. Denoised Difference Map ──────────────────────────────
    diff_map = cv2.absdiff(detail, denoised)
    fig2, ax2 = plt.subplots(figsize=(4,4))
    ax2.imshow(diff_map, cmap='inferno')
    ax2.set_title("Abs(Difference) Detail − Denoised")
    ax2.axis('off')
    st.pyplot(fig2)
