import streamlit as st
import cv2
import numpy as np
from math import sqrt, cos, sin, pi, atan2
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Compute the Edge Confidence Map
# -------------------------------
def compute_gradients(gray):
    """
    Compute gradient magnitude and orientation using Sobel operators.
    M(x,y) = sqrt(Gx^2 + Gy^2) serves as the edge confidence.
    Orientation is computed via arctan2(Gy, Gx) and wrapped to [0, 2*pi).
    """
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) % (2 * pi)
    return magnitude, orientation

# -------------------------------
# Step 2 & 3: Compute Block-Level DGC via Circular Variance
# -------------------------------
def block_dgc(mag_block, ori_block):
    """
    Compute circular variance for a block.
    σ = 1 - (resultant / total_weight)
    where:
      resultant = sqrt((Σ M*cosθ)^2 + (Σ M*sinθ)^2)
      total_weight = Σ M
    A low σ indicates consistent orientations.
    """
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    total_weight = np.sum(mag_block) + 1e-8  # avoid division by zero
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma

def compute_dgc(gray, block_size=8):
    """
    Divide the grayscale image into non-overlapping blocks and compute
    the DGC for each block using circular variance.
    Returns the local DGC map and the global DGC (average over blocks).
    """
    magnitude, orientation = compute_gradients(gray)
    rows, cols = gray.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size
    dgc_map = np.zeros((num_blocks_row, num_blocks_col))
    block_values = []
    
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            r0, r1 = i * block_size, i * block_size + block_size
            c0, c1 = j * block_size, j * block_size + block_size
            mag_block = magnitude[r0:r1, c0:c1]
            ori_block = orientation[r0:r1, c0:c1]
            sigma = block_dgc(mag_block, ori_block)
            dgc_map[i, j] = sigma
            block_values.append(sigma)
    global_dgc = np.mean(block_values)
    return dgc_map, global_dgc

# -------------------------------
# Streamlit App
# -------------------------------
def main():
    st.title("Directional Gradient Consistency (DGC) Metric")
    st.write("""
        This app computes a simplified DGC metric to quantify edge orientation consistency in an image.
        The metric uses:
        - An edge confidence map (via gradient magnitude).
        - Edge orientations from Sobel gradients.
        - Block-wise circular variance to measure the dispersion of these orientations.
        A higher global DGC value indicates greater disruption (potentially from steganographic embedding).
    """)
    
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error loading image.")
            return
        st.image(img, caption="Uploaded Image", use_column_width=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        block_size = st.slider("Select Block Size (for local DGC computation)", min_value=4, max_value=32, value=8, step=2)
        
        dgc_map, global_dgc = compute_dgc(gray, block_size)
        st.write(f"**Global DGC Metric:** {global_dgc:.4f}")
        
        # Display local DGC heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(dgc_map, cmap='hot', interpolation='nearest')
        ax.set_title("Local DGC per Block")
        fig.colorbar(cax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
