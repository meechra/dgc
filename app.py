import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt

st.title("DGC Score: Tunable Exponent, Weighting & Threshold")

# ─── Sidebar sliders for experimentation ────────────────────────────────
p = st.sidebar.slider(
    "Block DGC exponent (p)",
    min_value=1.0,
    max_value=5.0,
    value=2.5,
    step=0.1,
    help="Higher p → noisy blocks get amplified"
)

weight_exp = st.sidebar.slider(
    "Block weight exponent",
    min_value=1,
    max_value=3,
    value=2,
    step=1,
    help="1 = sum(mag), 2 = sum(mag²), 3 = sum(mag³)"
)

grad_threshold = st.sidebar.slider(
    "Gradient‑energy threshold",
    min_value=0.0,
    max_value=1000.0,
    value=1.0,
    step=1.0,
    help="Ignore blocks whose weight < threshold"
)

# ─── 1. Compute Sobel gradients → magnitude & orientation ──────────────
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ori = np.arctan2(Gy, Gx) % (2*pi)
    return mag, ori

# ─── 2. Block‐level circular‐variance DGC with tunable exponent ───────
def block_dgc(mag_block, ori_block, exponent):
    w = np.sum(mag_block) + 1e-8
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    R = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (R / w)
    return sigma ** exponent

# ─── 3. Weighted sliding‐window DGC with tunable weight & threshold ──
def compute_weighted_dgc_score(img, block_size=7, grad_th=1.0, w_exp=2, dgcp=2.5):
    mag, ori = compute_gradients(img)
    h, w = img.shape
    num = 0.0
    den = 0.0

    for i in range(0, h - block_size + 1, block_size):
        for j in range(0, w - block_size + 1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]

            # block weight = sum(mag_block ** weight_exp)
            block_weight = np.sum(mb ** w_exp)
            if block_weight < grad_th:
                continue

            dgc_val = block_dgc(mb, ob, dgcp)
            num += dgc_val * block_weight
            den += block_weight

    return (num / den) if den > 0 else 0.0

# ─── Wavelet detail extraction (as before) ─────────────────────────────
def get_wavelet_detail_image(gray):
    cA, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# ─── Denoise with bilateral filter (static for now) ────────────────────
def get_wavelet_denoised_image(gray):
    detail = get_wavelet_detail_image(gray)
    return cv2.bilateralFilter(detail, d=9, sigmaColor=75, sigmaSpace=75)

# ─── Simple bar chart to compare scores ────────────────────────────────
def plot_score_comparison(raw, denoised):
    fig, ax = plt.subplots()
    ax.bar(['Raw','Denoised'], [raw, denoised], color=['steelblue','orange'])
    ax.set_ylabel('Weighted DGC Score')
    ax.set_title('Raw vs. Denoised DGC')
    ax.set_ylim(0, max(raw, denoised)*1.1)
    return fig

# ─── Streamlit UI: Upload + processing ────────────────────────────────
uploaded = st.file_uploader("Upload a Grayscale Image", type=['png','jpg','jpeg'])
if uploaded:
    data = np.frombuffer(uploaded.read(), np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Could not decode image.")
    else:
        st.image(gray, caption="Original Grayscale", clamp=True, width=300)

        detail_img   = get_wavelet_detail_image(gray)
        denoised_img = get_wavelet_denoised_image(gray)

        st.image(detail_img,   caption="Wavelet Detail",   clamp=True, width=300)
        st.image(denoised_img, caption="Denoised Detail", clamp=True, width=300)

        raw_score = compute_weighted_dgc_score(
            detail_img,
            block_size=7,
            grad_th=grad_threshold,
            w_exp=weight_exp,
            dgcp=p
        )
        den_score = compute_weighted_dgc_score(
            denoised_img,
            block_size=7,
            grad_th=grad_threshold,
            w_exp=weight_exp,
            dgcp=p
        )
        diff = raw_score - den_score

        st.markdown(f"**Raw DGC Score:** {raw_score:.4f}")
        st.markdown(f"**Denoised DGC Score:** {den_score:.4f}")
        st.markdown(f"**Difference (raw − denoised):** {diff:.4f}")

        fig = plot_score_comparison(raw_score, den_score)
        st.pyplot(fig)

        st.write("""
        _Use the sliders in the sidebar to tweak:_  
        - **p** : exponent in block DGC  
        - **weight_exp** : edge‑energy weighting exponent  
        - **grad_threshold** : minimum block weight to include  
        """)
