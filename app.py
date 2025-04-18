import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt

# 1. Compute Sobel gradients → magnitude & orientation
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    mag = np.sqrt(Gx**2 + Gy**2)
    ori = (np.arctan2(Gy, Gx) % (2*pi))
    return mag, ori

# 2. Block‐level circular‐variance DGC
def block_dgc(mag_block, ori_block):
    w = np.sum(mag_block) + 1e-8
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    R = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (R / w)
    return sigma**1.5

# 3. Weighted sliding‐window DGC score
def compute_weighted_dgc_score(img, block_size=7, grad_threshold=1.0):
    mag, ori = compute_gradients(img)
    h, w = img.shape
    num = 0.0
    den = 0.0
    for i in range(0, h-block_size+1, block_size):
        for j in range(0, w-block_size+1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]
            weight = np.sum(mb)
            if weight < grad_threshold:
                continue
            num += block_dgc(mb, ob) * weight
            den += weight
    return (num/den) if den>0 else 0.0

# 4a. Wavelet‐detail extraction (high‑freq)
def get_wavelet_detail_image(gray):
    cA, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    return cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 4b. Wavelet soft‑threshold denoising
def get_wavelet_denoised_image(gray):
    arr = gray.astype(np.float32)
    coeffs = pywt.wavedec2(arr, 'db1', level=1)
    cA, (cH, cV, cD) = coeffs
    # Universal threshold (Donoho)
    sigma = np.median(np.abs(cD)) / 0.6745
    uth = sigma * np.sqrt(2*np.log(arr.size))
    cH = pywt.threshold(cH, uth, mode='soft')
    cV = pywt.threshold(cV, uth, mode='soft')
    cD = pywt.threshold(cD, uth, mode='soft')
    den_coeffs = [cA, (cH, cV, cD)]
    denoised = pywt.waverec2(den_coeffs, 'db1')
    return cv2.normalize(denoised, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 5. Simple bar chart comparison
def plot_score_comparison(raw, den):
    fig, ax = plt.subplots()
    ax.bar(['Detail','Denoised'], [raw, den], color=['steelblue','orange'])
    ax.set_ylabel('Weighted DGC Score')
    ax.set_title('Raw vs. Wavelet‑Denoised DGC')
    ax.set_ylim(0, max(raw, den)*1.1)
    return fig

# --- Streamlit UI ---
st.title("Wavelet‑Detail DGC: Raw vs. Wavelet‑Denoised")

uploaded = st.file_uploader("Upload Grayscale Image", type=['png','jpg','jpeg'])
if uploaded:
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Couldn't decode the image.")
    else:
        st.image(gray, caption="Original Grayscale", clamp=True, width=300)

        # Detail & Denoised images
        detail_img   = get_wavelet_detail_image(gray)
        denoised_img = get_wavelet_denoised_image(gray)

        st.image(detail_img, caption="Wavelet Detail", clamp=True, width=300)
        st.image(denoised_img, caption="Wavelet‑Denoised", clamp=True, width=300)

        # Compute scores
        raw_score  = compute_weighted_dgc_score(detail_img)
        den_score  = compute_weighted_dgc_score(denoised_img)
        diff       = raw_score - den_score

        st.markdown(f"**Raw DGC Score:** {raw_score:.4f}")
        st.markdown(f"**Denoised DGC Score:** {den_score:.4f}")
        st.markdown(f"**Difference (raw − denoised):** {diff:.4f}")

        # Bar chart
        fig = plot_score_comparison(raw_score, den_score)
        st.pyplot(fig)

        st.write("""
        _A larger gap (difference) means wavelet denoising removed more high‑freq directional inconsistency  
        — i.e. stronger stego‑like artifacts in the detail image._
        """)
