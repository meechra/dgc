import streamlit as st
import cv2
import numpy as np
from math import sqrt, pi
from io import BytesIO
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

def normalize_difference(diff, min_diff=-0.002356, max_diff=0.039568):
    """Normalize the difference to [0,1] range."""
    return (diff - min_diff) / (max_diff - min_diff)

def read_image(uploaded_file):
    """Read an uploaded image and convert to grayscale."""
    file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
    image = cv2.imdecode(file_bytes, cv2.IMREAD_GRAYSCALE)
    return image

def plot_interference_line(normalized_diff):
    """
    Plot a number line from 0 to 1 with markers indicating interference.
    The left end is labeled "0 interference" and the right end "max interference".
    """
    fig, ax = plt.subplots(figsize=(8, 2))
    # Plot the line
    ax.hlines(0, 0, 1, colors='gray', linewidth=4)
    # Plot the marker for the normalized difference
    ax.plot(normalized_diff, 0, marker='o', markersize=12, color='red')
    # Set labels for the extremes
    ax.text(0, 0.1, '0 interference', ha='left', va='center', fontsize=10, color='black')
    ax.text(1, 0.1, 'max interference', ha='right', va='center', fontsize=10, color='black')
    # Remove y-axis and spines for cleaner look
    ax.get_yaxis().set_visible(False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    # Set limits and title
    ax.set_xlim(-0.05, 1.05)
    ax.set_title("Stego Interference Indicator")
    return fig

# --- Streamlit App ---

st.title("DGC Score Calculator with Interference Indicator")
st.write("""
This app calculates the Directional Gradient Consistency (DGC) scores for a pair of images:
- A **Clean** image
- A **Stego** (steganographically modified) image

The computation divides each image into 7×7 blocks and computes a global DGC score as the average of local scores.
The difference between the stego and clean scores is then normalized to a [0,1] range, which is visualized on a number line:
- **0 interference** on the left
- **max interference** on the right
Upload both images below to compute and compare their DGC scores.
""")

# File upload widgets for the clean and stego images
clean_file = st.file_uploader("Upload Clean Image", type=["png", "jpg", "jpeg"])
stego_file = st.file_uploader("Upload Stego Image", type=["png", "jpg", "jpeg"])

if clean_file is not None and stego_file is not None:
    # Read the images
    clean_img = read_image(clean_file)
    stego_img = read_image(stego_file)
    
    if clean_img is None or stego_img is None:
        st.error("Error reading one or both images. Please try a different file.")
    else:
        # Optionally display the images
        st.image([clean_img, stego_img], caption=["Clean Image", "Stego Image"], width=250)

        # Denoise images
        clean_img_denoised = denoise_gray(clean_img)
        stego_img_denoised = denoise_gray(stego_img)
        
        # Compute global DGC scores using 7x7 blocks
        clean_global = compute_global_dgc(clean_img_denoised, block_size=7)
        stego_global = compute_global_dgc(stego_img_denoised, block_size=7)
        
        # Compute the difference and normalized difference
        diff = stego_global - clean_global
        normalized_diff = normalize_difference(diff)
        
        # Display the results
        st.write("### Global DGC Scores:")
        st.write(f"**Clean DGC Score:** {clean_global:.6f}")
        st.write(f"**Stego DGC Score:** {stego_global:.6f}")
        st.write(f"**Raw Difference (Stego - Clean):** {diff:.6f}")
        st.write(f"**Normalized Difference:** {normalized_diff:.6f}")
        
        # Plot and display the interference number line
        interference_fig = plot_interference_line(normalized_diff)
        st.pyplot(interference_fig)
