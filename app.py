import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi

# --- 1. Gradient Computation ---
def compute_gradients(gray):
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude   = np.sqrt(Gx**2 + Gy**2)
    orientation = (np.arctan2(Gy, Gx) % (2 * pi))
    return magnitude, orientation

# --- 2. Block‐level DGC ---
def block_dgc(mag_block, ori_block):
    total_weight = np.sum(mag_block) + 1e-8
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma ** 1.5

# --- 3. Weighted sliding‐window DGC ---
def compute_weighted_dgc_score(image, block_size=7, grad_threshold=1.0):
    mag, ori = compute_gradients(image)
    rows, cols = image.shape
    weighted_sum = 0.0
    total_weight = 0.0

    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]
            w = np.sum(mb)
            if w < grad_threshold:
                continue
            weighted_sum += block_dgc(mb, ob) * w
            total_weight += w

    return (weighted_sum / total_weight) if total_weight > 0 else 0.0

# --- 4. Wavelet detail extraction ---
def get_wavelet_detail_image(gray):
    cA, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    detail_norm = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    return detail_norm.astype(np.uint8)

# --- 5. Normalize DGC to [0,1] ---
def normalize_score(score, min_score=0.0, max_score=1.0):
    return np.clip((score - min_score) / (max_score - min_score), 0.0, 1.0)

# --- Streamlit UI ---
st.title("Wavelet‐Detail DGC Score with Normalization")

uploaded = st.file_uploader("Upload Grayscale Image", type=['png','jpg','jpeg'])
if uploaded:
    # Read image
    data = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Could not decode image.")
    else:
        st.image(gray, caption="Original Image", clamp=True, channels="GRAY", width=300)

        # Wavelet‐detail
        detail_img = get_wavelet_detail_image(gray)
        st.image(detail_img, caption="Wavelet Detail Image", clamp=True, channels="GRAY", width=300)

        # DGC scoring
        raw_score  = compute_weighted_dgc_score(detail_img)
        norm_score = normalize_score(raw_score, min_score=0.0, max_score=1.0)

        st.markdown(f"**Raw DGC Score:** {raw_score:.4f}")
        st.markdown(f"**Normalized DGC Score:** {norm_score:.4f}")
        st.write("""
        _Higher normalized scores (closer to 1) → more directional incoherence  
        (i.e. possible stego distortion). Lower (closer to 0) → strong coherence (clean)._
        """)
