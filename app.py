import streamlit as st
import cv2
import numpy as np
import pywt
from math import sqrt, pi
import matplotlib.pyplot as plt

# --- Core DGC Functions ---

def compute_gradients(gray):
    """Compute gradient magnitude and orientation using Sobel."""
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) % (2 * pi)
    return magnitude, orientation


def block_dgc(mag_block, ori_block):
    """Compute DGC score for a block (circular variance)."""
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    total_weight = np.sum(mag_block) + 1e-8
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma ** 1.5


def compute_weighted_dgc_score(gray, block_size=7, grad_threshold=1e-3):
    """
    Compute weighted average DGC over image blocks,
    skipping low-energy regions.
    """
    mag, ori = compute_gradients(gray)
    rows, cols = gray.shape
    total_score = 0.0
    total_weight = 0.0

    for i in range(0, rows - block_size + 1, block_size):
        for j in range(0, cols - block_size + 1, block_size):
            mb = mag[i:i+block_size, j:j+block_size]
            ob = ori[i:i+block_size, j:j+block_size]
            weight = np.sum(mb)
            if weight < grad_threshold:
                continue
            score = block_dgc(mb, ob)
            total_score += score * weight
            total_weight += weight

    return (total_score / total_weight) if total_weight > 0 else 0.0

# --- Wavelet Denoising ---

def wavelet_denoise(gray, wavelet='db1', level=1):
    """
    Perform single-level wavelet denoising using soft thresholding.
    """
    coeffs = pywt.wavedec2(gray.astype(np.float32), wavelet, level=level)
    cA, details = coeffs[0], coeffs[1]
    cH, cV, cD = details
    # Estimate noise sigma from diagonal detail
    sigma = np.median(np.abs(cD)) / 0.6745
    uthresh = sigma * np.sqrt(2 * np.log(gray.size))
    cH_t = pywt.threshold(cH, uthresh, mode='soft')
    cV_t = pywt.threshold(cV, uthresh, mode='soft')
    cD_t = pywt.threshold(cD, uthresh, mode='soft')
    coeffs_denoised = [cA, (cH_t, cV_t, cD_t)]
    denoised = pywt.waverec2(coeffs_denoised, wavelet)
    denoised = np.clip(denoised, 0, 255)
    return denoised.astype(np.uint8)

# --- Utility & Visualization ---

def normalize_difference(diff, min_diff=-0.002356, max_diff=0.039568):
    return (diff - min_diff) / (max_diff - min_diff)


def plot_metric_line(norm_diff):
    fig, ax = plt.subplots(figsize=(8, 2))
    ax.hlines(0, 0, 1, colors='gray', linewidth=4)
    ax.plot(norm_diff, 0, 'o', markersize=12, color='red')
    ax.text(norm_diff, 0.1, f"{norm_diff:.3f}", ha='center', va='bottom')
    ax.text(0, -0.1, 'no change', ha='left', va='top')
    ax.text(1, -0.1, 'max change', ha='right', va='top')
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values(): spine.set_visible(False)
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title("Normalized DGC Change")
    return fig

# --- Streamlit App ---
st.title("DGC: Original vs Wavelet-Denoised")
st.write(
    "Upload an image to compare its DGC score against a wavelet-denoised version."
)

uploaded = st.file_uploader("Upload Image", type=["png","jpg","jpeg"])
if uploaded:
    data = np.frombuffer(uploaded.read(), dtype=np.uint8)
    gray = cv2.imdecode(data, cv2.IMREAD_GRAYSCALE)
    if gray is None:
        st.error("Could not read image. Try another file.")
    else:
        denoised = wavelet_denoise(gray)
        orig_score = compute_weighted_dgc_score(gray)
        denoised_score = compute_weighted_dgc_score(denoised)
        diff = orig_score - denoised_score
        norm_diff = normalize_difference(diff)

        st.image([gray, denoised], caption=["Original","Denoised"], width=250)
        st.write(f"**Original DGC Score:** {orig_score:.4f}")
        st.write(f"**Denoised DGC Score:** {denoised_score:.4f}")
        st.write(f"**Normalized Change:** {norm_diff:.3f}")

        fig = plot_metric_line(norm_diff)
        st.pyplot(fig)
