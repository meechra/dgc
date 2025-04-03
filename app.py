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
    Since each block's DGC is in [0,1] (with 0 indicating perfect alignment and 1 indicating random orientations),
    the global score (the average) is also in [0,1].
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

def read_image(uploaded_file):
    """Read an uploaded image and convert to grayscale."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return image

def plot_gap_line(clean_score, stego_score):
    """
    Plot a horizontal number line from 0 to 1 with the following:
      - A marker for the clean image's global DGC score.
      - A marker for the stego image's global DGC score.
    The left end is labeled "0 interference" and the right end "max interference".
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    
    # Draw the horizontal line.
    ax.hlines(0, 0, 1, colors='gray', linewidth=4)
    
    # Plot the markers for the clean and stego scores.
    ax.plot(clean_score, 0, marker='o', markersize=12, color='blue', label='Clean')
    ax.plot(stego_score, 0, marker='o', markersize=12, color='red', label='Stego')
    
    # Add text labels above the markers.
    ax.text(clean_score, 0.1, f"Clean: {clean_score:.3f}", ha='center', va='bottom', fontsize=10, color='blue')
    ax.text(stego_score, 0.1, f"Stego: {stego_score:.3f}", ha='center', va='bottom', fontsize=10, color='red')
    
    # Set labels for the extremes.
    ax.text(0, -0.1, '0 interference', ha='left', va='top', fontsize=10, color='black')
    ax.text(1, -0.1, 'max interference', ha='right', va='top', fontsize=10, color='black')
    
    # Remove y-axis and extra spines for a cleaner look.
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    
    ax.set_xlim(-0.05, 1.05)
    ax.set_ylim(-0.3, 0.3)
    ax.set_title("Stego Interference Indicator")
    ax.legend(loc='upper center', bbox_to_anchor=(0.5, 1.25), ncol=2)
    return fig

# --- Streamlit App ---

st.title("DGC Score Calculator with Gap Visualization")
st.write("""
This app calculates the Directional Gradient Consistency (DGC) scores for a pair of images:
- A **Clean** image
- A **Stego** (steganographically modified) image

Each image is divided into 7×7 blocks, and a global DGC score is computed as the average of local block scores.
A lower DGC score indicates less interference, while a higher score indicates more interference.
The number line below shows both the clean and stego scores on a scale from 0 (0 interference) to 1 (max interference).
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

        # Denoise images.
        clean_img_denoised = denoise_gray(clean_img)
        stego_img_denoised = denoise_gray(stego_img)
        
        # Compute global DGC scores using 7x7 blocks.
        clean_global = compute_global_dgc(clean_img_denoised, block_size=7)
        stego_global = compute_global_dgc(stego_img_denoised, block_size=7)
        
        # Compute the difference.
        diff = stego_global - clean_global
        
        # Display the computed scores and the difference.
        st.write("### Global DGC Scores:")
        st.write(f"**Clean DGC Score:** {clean_global:.6f}")
        st.write(f"**Stego DGC Score:** {stego_global:.6f}")
        st.write(f"**Raw Difference (Stego - Clean):** {diff:.6f}")
        
        # Plot and display the number line with both scores.
        gap_fig = plot_gap_line(clean_global, stego_global)
        st.pyplot(gap_fig)
