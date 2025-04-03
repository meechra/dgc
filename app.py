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
    """Compute gradients using Sobel operator."""
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

def normalize_score(score, min_score=0.0, max_score=1.0):
    """
    Normalize a score to [0,1] using calibration values.
    Update min_score and max_score if you have different calibration values.
    """
    return (score - min_score) / (max_score - min_score)

def read_image(uploaded_file):
    """Read an uploaded image and convert to grayscale."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return image

def plot_gap_line(norm_clean, norm_stego):
    """
    Plot a horizontal number line from 0 to 100 with markers indicating:
      - The normalized clean DGC score.
      - The normalized stego DGC score.
    The left end is labeled "0 interference" and the right end "max interference".
    The scores are scaled by 100 to magnify differences (2 decimal points).
    """
    # Multiply normalized values by 100 for visualization.
    vis_clean = norm_clean * 100
    vis_stego = norm_stego * 100
    
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Draw the horizontal baseline.
    ax.hlines(0, 0, 100, colors='gray', linewidth=4)
    
    # Plot markers for normalized scores.
    ax.plot(vis_clean, 0, marker='o', markersize=12, color='blue', label='Clean')
    ax.plot(vis_stego, 0, marker='o', markersize=12, color='red', label='Stego')
    
    # Add text labels above the markers.
    ax.text(vis_clean, 0.1, f"Clean: {vis_clean:.2f}", ha='center', va='bottom', fontsize=10, color='blue')
    ax.text(vis_stego, 0.1, f"Stego: {vis_stego:.2f}", ha='center', va='bottom', fontsize=10, color='red')
    
    # Label the extremes.
    ax.text(0, -0.1, '0 interference', ha='left', va='top', fontsize=10, color='black')
    ax.text(100, -0.1, 'max interference', ha='right', va='top', fontsize=10, color='black')
    
    # Clean up the plot.
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.set_xlim(-5, 105)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title("Stego Interference Indicator (Normalized Scores)")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    return fig

# --- Streamlit App ---

st.title("DGC Score Calculator with Magnified Gap Visualization")
st.write("""
This app calculates the Directional Gradient Consistency (DGC) scores for a pair of images:
- A **Clean** image
- A **Stego** (steganographically modified) image

Each image is divided into 7×7 blocks, and a global DGC score is computed as the average of the local block scores.
We then normalize these scores (using the calibration range of [0,1]) and magnify them by multiplying by 100,
so that the output represents a normalized interference level with a finer scale.
Only the normalized scores (in the 0-100 range) are displayed.
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
        
        # Normalize the scores.
        norm_clean = normalize_score(clean_global, min_score=0.0, max_score=1.0)
        norm_stego = normalize_score(stego_global, min_score=0.0, max_score=1.0)
        
        # Display only the normalized scores (magnified).
        st.write("### Normalized Global DGC Scores (Magnified):")
        st.write(f"**Normalized Clean DGC Score:** {(norm_clean*100):.2f}")
        st.write(f"**Normalized Stego DGC Score:** {(norm_stego*100):.2f}")
        
        # Plot and display the magnified number line visualization.
        gap_fig = plot_gap_line(norm_clean, norm_stego)
        st.pyplot(gap_fig)
