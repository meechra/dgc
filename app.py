import streamlit as st
import cv2
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt

# --- Helper Functions ---

def denoise_gray(gray):
    """Denoise the grayscale image."""
    return cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)

def compute_gradients(gray):
    """Compute gradients using the Sobel operator."""
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) % (2 * pi)
    return magnitude, orientation

def block_dgc(mag_block, ori_block):
    """Compute weighted circular variance for a block."""
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    total_weight = np.sum(mag_block) + 1e-8  # Prevent division by zero
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma ** 1.5  # Non-linear scaling

def compute_global_dgc(gray, block_size=7):
    """
    Compute the global DGC score for a grayscale image using blocks of size block_size x block_size.
    """
    magnitude, orientation = compute_gradients(gray)
    rows, cols = gray.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size

    dgc_values = []
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            r0, r1 = i * block_size, (i + 1) * block_size
            c0, c1 = j * block_size, (j + 1) * block_size
            mag_block = magnitude[r0:r1, c0:c1]
            ori_block = orientation[r0:r1, c0:c1]
            dgc_values.append(block_dgc(mag_block, ori_block))
    
    return np.mean(dgc_values)

def normalize_difference(diff, min_diff=-0.002356, max_diff=0.039568):
    """
    Normalize the difference to a [0,1] range using calibration values.
    Here, diff = stego_global - clean_global.
    """
    return (diff - min_diff) / (max_diff - min_diff)

def read_image(uploaded_file):
    """Read an uploaded image and convert it to grayscale."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return image

def plot_metric_line(norm_diff):
    """
    Plot a horizontal number line from 0 to 1 with a marker at the normalized difference.
    Label the left end as "no interference" and the right end as "max interference."
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Draw a horizontal baseline across the full [0,1] range.
    ax.hlines(0, 0, 1, colors='gray', linewidth=4)
    
    # Plot the marker for the normalized difference.
    ax.plot(norm_diff, 0, marker='o', markersize=12, color='red')
    
    # Add text label above the marker.
    ax.text(norm_diff, 0.1, f"{norm_diff:.3f}", ha='center', va='bottom', fontsize=10, color='red')
    
    # Label the extremities.
    ax.text(0, -0.1, 'min diff', ha='left', va='top', fontsize=10, color='black')
    ax.text(1, -0.1, 'max diff', ha='right', va='top', fontsize=10, color='black')
    
    # Clean up the plot.
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title("Normalized Difference Metric")
    return fig

# --- Streamlit App ---

st.title("DGC Score Calculator with Normalized Difference Metric")
st.write("""
This app calculates the Directional Gradient Consistency (DGC) scores for a pair of images:
- A **Clean** image
- A **Stego** (steganographically modified) image

Each image is divided into 7×7 blocks, and a global DGC score is computed as the average of the local block scores.
The final metric is the normalized difference between the stego and clean scores,
normalized to a [0,1] range using calibration values.
""")

# File upload widgets for the clean and stego images.
clean_file = st.file_uploader("Upload Clean Image", type=["png", "jpg", "jpeg"])
stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])

if clean_file is not None and stego_file is not None:
    # Read the images.
    clean_img = read_image(clean_file)
    stego_img = read_image(stego_file)
    
    if clean_img is None or stego_img is None:
        st.error("Error reading one or both images. Please try a different file.")
    else:
        # Optionally display the images.
        st.image([clean_img, stego_img], caption=["Clean Image", "Stego Image"], width=250)

        # Denoise the images.
        clean_img_denoised = denoise_gray(clean_img)
        stego_img_denoised = denoise_gray(stego_img)
        
        # Compute the global DGC scores using 7x7 blocks.
        clean_global = compute_global_dgc(clean_img_denoised, block_size=7)
        stego_global = compute_global_dgc(stego_img_denoised, block_size=7)
        
        # Compute the difference (stego - clean) and normalize it.
        diff = stego_global - clean_global
        norm_diff = normalize_difference(diff, min_diff=-0.002356, max_diff=0.039568)
        
        # Display only the normalized difference metric.
        st.write("### Normalized Difference Metric:")
        st.write(f"**Normalized Difference:** {norm_diff:.3f}")
        
        # Plot and display the number line visualization.
        metric_fig = plot_metric_line(norm_diff)
        st.pyplot(metric_fig)
