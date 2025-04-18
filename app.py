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

# --- 2. Block‐level DGC (circular variance) ---
def block_dgc(mag_block, ori_block):
    total_weight = np.sum(mag_block) + 1e-8
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma ** 1.5  # non‐linear scaling as before

# --- 3. Weighted, thresholded, sliding‐window DGC ---
def compute_weighted_dgc_score(image, block_size=7, grad_threshold=1.0):
    mag, ori = compute_gradients(image)
    rows, cols = image.shape
    weighted_sum = 0.0
    total_weight = 0.0

    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            mag_block = mag[i:i+block_size, j:j+block_size]
            ori_block = ori[i:i+block_size, j:j+block_size]
            block_weight = np.sum(mag_block)
            if block_weight < grad_threshold:
                continue
            dgc = block_dgc(mag_block, ori_block)
            weighted_sum   += dgc * block_weight
            total_weight   += block_weight

    if total_weight == 0:
        return 0.0
    return weighted_sum / total_weight

# --- 4. Wavelet‐detail extraction (single‐level Haar) ---
def get_wavelet_detail_image(gray):
    # Perform 1‐level DWT
    cA, (cH, cV, cD) = pywt.dwt2(gray.astype(np.float32), 'db1')
    detail = np.sqrt(cH**2 + cV**2 + cD**2)
    # Normalize to [0,255]
    detail_norm = cv2.normalize(detail, None, 0, 255, cv2.NORM_MINMAX)
    return detail_norm.astype(np.uint8)

# --- 5. One‐step score: wavelet‐detail + weighted DGC ---
def get_dgc_wavelet_detail_score(image):
    detail_img = get_wavelet_detail_image(image)
    return compute_weighted_dgc_score(detail_img)

# --- Streamlit UI ---
st.title("DGC Wavelet‐Detail Score Calculator")

uploaded = st.file_uploader("Upload a Grayscale Image", type=['png','jpg','jpeg'])
if uploaded is not None:
    # Read and convert to grayscale numpy array
    file_bytes = np.asarray(bytearray(uploaded.read()), dtype=np.uint8)
    gray = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Could not read image.")
    else:
        st.image(gray, caption="Input Image", clamp=True, channels="GRAY", width=300)

        score = get_dgc_wavelet_detail_score(gray)
        st.markdown(f"**DGC Wavelet‐Detail Score:** {score:.4f}")

        st.write("""
        _Higher scores → less directional coherence in edge‐rich (wavelet‐detail) regions  
        (i.e. more likely “stego” distortion)._  
        Lower scores → strong coherence (i.e. likely clean).  
        """)
