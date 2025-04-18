import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt
from scipy.stats import entropy

st.set_page_config(layout="wide")
st.title("Fused Stego‑Interference Detector")

# ─── Sidebar: Tunable Parameters ────────────────────────────────────────
with st.sidebar:
    st.header("DGC Parameters")
    p                = st.slider("Block DGC exponent (p)", 1.0, 5.0, 2.5, 0.1)
    weight_exp       = st.slider("Block weight exponent", 1, 3, 2, 1)
    grad_threshold   = st.slider("Gradient‑energy threshold", 0.0, 1000.0, 1.0, 1.0)

    st.header("Multi‑Scale Settings")
    scales           = st.multiselect(
        "Block sizes for multi‑scale DGC",
        options=[7, 14, 28],
        default=[7, 14, 28]
    )
    stride_ms        = st.slider("Multi‑Scale stride", 1, 14, 7, 1)

    st.header("Histogram Divergence")
    bins             = st.slider("Orientation histogram bins", 4, 32, 16, 2)
    stride_hd        = st.slider("Hist‑div stride", 1, 14, 7, 1)

    st.header("Fusion Weight")
    alpha            = st.slider("α (DGC vs. Hist‑div)", 0.0, 1.0, 0.5, 0.05)

# ─── 1. Sobel gradients → magnitude & orientation ───────────────────────
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ori = np.arctan2(Gy, Gx) % (2*pi)
    return mag, ori

# ─── 2. Block DGC (circular variance) ─────────────────────────────────
def block_dgc(mb, ob, exponent):
    w = np.sum(mb) + 1e-8
    sum_cos = np.sum(mb * np.cos(ob))
    sum_sin = np.sum(mb * np.sin(ob))
    R = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (R / w)
    return sigma ** exponent

# ─── 3. Weighted DGC for a single block_size (non‑overlapping) ────────
def compute_weighted_dgc_score(img, block_size, grad_th, w_exp, dgcp):
    mag, ori = compute_gradients(img)
    h, w = img.shape
    num = den = 0.0
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]
            weight = np.sum(mb ** w_exp)
            if weight < grad_th:
                continue
            num += block_dgc(mb, ob, dgcp) * weight
            den += weight
    return (num/den) if den>0 else 0.0

# ─── 4. Multi‑Scale DGC ────────────────────────────────────────────────
def multiscale_dgc(detail_img, scales, stride, grad_th, w_exp, dgcp):
    scores = []
    mag, ori = compute_gradients(detail_img)
    H, W = detail_img.shape

    for bs in scales:
        vals = []
        for i in range(0, H-bs+1, stride):
            for j in range(0, W-bs+1, stride):
                mb = mag[i:i+bs, j:j+bs]
                ob = ori[i:i+bs, j:j+bs]
                weight = np.sum(mb ** w_exp)
                if weight < grad_th:
                    continue
                vals.append(block_dgc(mb, ob, dgcp))
        if vals:
            scores.append(np.mean(vals))
    return max(scores) if scores else 0.0

# ─── 5a. Wavelet‑Detail Extraction ────────────────────────────────────
def get_wavelet_detail_image(gray):
    _, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ─── 5b. Denoising via Bilateral Filter on Detail ─────────────────────
def get_wavelet_denoised_image(detail_img):
    return cv2.bilateralFilter(detail_img, d=9, sigmaColor=75, sigmaSpace=75)

# ─── 6. Orientation‑Histogram Divergence ─────────────────────────────
def hist_divergence(detail_img, denoised_img, bins, block_size, stride):
    mag1, ori1 = compute_gradients(detail_img)
    mag2, ori2 = compute_gradients(denoised_img)
    kl_vals = []
    for i in range(0, ori1.shape[0]-block_size+1, stride):
        for j in range(0, ori1.shape[1]-block_size+1, stride):
            o1 = ori1[i:i+block_size, j:j+block_size].ravel()
            w1 = mag1[i:i+block_size, j:j+block_size].ravel()
            o2 = ori2[i:i+block_size, j:j+block_size].ravel()
            w2 = mag2[i:i+block_size, j:j+block_size].ravel()
            h1, _ = np.histogram(o1, bins=bins, range=(0,2*pi), weights=w1, density=True)
            h2, _ = np.histogram(o2, bins=bins, range=(0,2*pi), weights=w2, density=True)
            h1 += 1e-8; h2 += 1e-8
            kl_vals.append(entropy(h1, h2))
    return np.mean(kl_vals) if kl_vals else 0.0

# ─── Utility: Bar chart ───────────────────────────────────────────────
def plot_bar(labels, values, title, ylim=None):
    fig, ax = plt.subplots()
    ax.bar(labels, values, color=['steelblue','orange','green'])
    ax.set_title(title)
    ax.set_ylim(0, ylim or max(values)*1.1)
    return fig

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
        st.image(detail_img,   caption="Wavelet Detail",      width=250)
        st.image(denoised_img, caption="Denoised Detail",     width=250)

        # Compute features
        ms_score = multiscale_dgc(
            detail_img, scales, stride_ms,
            grad_threshold, weight_exp, p
        )
        hd_score = hist_divergence(
            detail_img, denoised_img,
            bins, block_size=7, stride=stride_hd
        )

        # Normalize to [0,1] by their max
        norm_base = max(ms_score, hd_score, 1e-8)
        ms_norm   = ms_score / norm_base
        hd_norm   = hd_score / norm_base

        fused     = alpha * ms_norm + (1-alpha) * hd_norm

        # Show scores
        st.markdown(f"**Multi‑Scale DGC:** {ms_score:.4f} (normalized {ms_norm:.4f})")
        st.markdown(f"**Hist‑Divergence:** {hd_score:.4f} (normalized {hd_norm:.4f})")
        st.markdown(f"**Fused Score (α={alpha}):** {fused:.4f}")

        # Plot bar chart
        fig = plot_bar(
            ['Multi‑Scale', 'Hist‑Div', 'Fused'],
            [ms_norm, hd_norm, fused],
            "Normalized Feature Contributions",
            ylim=1.0
        )
        st.pyplot(fig)
