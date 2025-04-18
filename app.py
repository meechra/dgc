import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
from scipy.stats import entropy

st.set_page_config(layout="wide")
st.title("Fused Stego‑Interference Detector")

# ─── Fixed DGC Parameters ───────────────────────────────────────────────
P_EXPONENT     = 2.5    # block_dgc exponent
WEIGHT_EXP     = 2      # block weight = sum(mag**WEIGHT_EXP)
GRAD_THRESHOLD = 1.0    # ignore blocks below this edge‑energy

# ─── Sidebar: Only Multi‑Scale, Hist‑Div & Fusion ───────────────────────
with st.sidebar:
    st.header("Multi‑Scale DGC")
    scales    = st.multiselect("Block sizes", [7, 14, 28], default=[7, 14, 28])
    stride_ms = st.slider("Multi‑Scale stride", 1, 14, 7, 1)

    st.header("Histogram Divergence")
    bins       = st.slider("Orientation histogram bins", 4, 32, 16, 2)
    stride_hd  = st.slider("Hist‑div stride", 1, 14, 7, 1)

    st.header("Fusion Weight")
    alpha      = st.slider("α (DGC vs. Hist‑div)", 0.0, 1.0, 0.5, 0.05)

# ─── 1. Sobel gradients → magnitude & orientation ───────────────────────
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ori = np.arctan2(Gy, Gx) % (2*pi)
    return mag, ori

# ─── 2. Block DGC (circular variance) ─────────────────────────────────
def block_dgc(mb, ob):
    w = np.sum(mb) + 1e-8
    sum_cos = np.sum(mb * np.cos(ob))
    sum_sin = np.sum(mb * np.sin(ob))
    R = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (R / w)
    return sigma ** P_EXPONENT

# ─── 3. Weighted DGC for a single block_size ───────────────────────────
def compute_weighted_dgc_score(img, block_size):
    mag, ori = compute_gradients(img)
    h, w = img.shape
    num = den = 0.0
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]
            weight = np.sum(mb ** WEIGHT_EXP)
            if weight < GRAD_THRESHOLD:
                continue
            num += block_dgc(mb, ob) * weight
            den += weight
    return (num/den) if den > 0 else 0.0

# ─── 4. Multi‑Scale DGC ────────────────────────────────────────────────
def multiscale_dgc(detail_img, scales, stride):
    mag, ori = compute_gradients(detail_img)
    H, W = detail_img.shape
    scale_scores = []
    for bs in scales:
        vals = []
        for i in range(0, H-bs+1, stride):
            for j in range(0, W-bs+1, stride):
                mb = mag[i:i+bs, j:j+bs]
                ob = ori[i:i+bs, j:j+bs]
                weight = np.sum(mb ** WEIGHT_EXP)
                if weight < GRAD_THRESHOLD:
                    continue
                vals.append(block_dgc(mb, ob))
        if vals:
            scale_scores.append(np.mean(vals))
    return max(scale_scores) if scale_scores else 0.0

# ─── 5a. Wavelet‑Detail Extraction ────────────────────────────────────
def get_wavelet_detail_image(gray):
    _, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ─── 5b. Denoising via Bilateral Filter on Detail ─────────────────────
def get_wavelet_denoised_image(detail_img):
    return cv2.bilateralFilter(detail_img, d=9, sigmaColor=75, sigmaSpace=75)

# ─── 6. Orientation‑Histogram Divergence (robust) ─────────────────────
def hist_divergence(detail_img, denoised_img, bins, block_size, stride):
    mag1, ori1 = compute_gradients(detail_img)
    mag2, ori2 = compute_gradients(denoised_img)
    kl_vals = []
    eps = 1e-8
    H, W = ori1.shape

    for i in range(0, H-block_size+1, stride):
        for j in range(0, W-block_size+1, stride):
            mb1 = mag1[i:i+block_size, j:j+block_size]
            ob1 = ori1[i:i+block_size, j:j+block_size]
            mb2 = mag2[i:i+block_size, j:j+block_size]
            ob2 = ori2[i:i+block_size, j:j+block_size]

            w1, w2 = mb1.sum(), mb2.sum()
            if w1 < GRAD_THRESHOLD or w2 < GRAD_THRESHOLD:
                continue

            h1, _ = np.histogram(ob1.ravel(), bins=bins, range=(0,2*pi), weights=mb1.ravel())
            h2, _ = np.histogram(ob2.ravel(), bins=bins, range=(0,2*pi), weights=mb2.ravel())

            p = h1 / (h1.sum() + eps)
            q = h2 / (h2.sum() + eps)
            p += eps; q += eps
            p /= p.sum(); q /= q.sum()

            kl_vals.append(entropy(p, q))

    return float(np.mean(kl_vals)) if kl_vals else 0.0

# ─── Streamlit App Body ──────────────────────────────────────────────
uploaded = st.file_uploader("Upload Grayscale Image", type=['png','jpg','jpeg'])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Invalid image.")
    else:
        # Display inputs
        st.image(gray, caption="Original Grayscale", width=250)
        detail_img   = get_wavelet_detail_image(gray)
        denoised_img = get_wavelet_denoised_image(detail_img)
        st.image(detail_img,   caption="Wavelet Detail",  width=250)
        st.image(denoised_img, caption="Denoised Detail", width=250)

        # Compute features
        ms_score = multiscale_dgc(detail_img, scales, stride_ms)
        hd_score = hist_divergence(detail_img, denoised_img, bins, block_size=7, stride=stride_hd)

        # Fuse directly (no normalization)
        fused     = alpha * ms_score + (1-alpha) * hd_score

        # Show scores
        st.markdown(f"**Multi‑Scale DGC:** {ms_score:.4f}")
        st.markdown(f"**Hist‑Divergence:** {hd_score:.4f}")
        st.markdown(f"**Fused Score (α={alpha}):** {fused:.4f}")

        # Likelihood indicator
        likelihood = fused * 100
        st.markdown(f"### Likelihood of Stego Interference: {likelihood:.1f}%")
