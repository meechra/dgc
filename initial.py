import streamlit as st
import cv2
import numpy as np
from math import sqrt, pi
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Compute the Edge Confidence Map
# -------------------------------
def compute_gradients(gray):
    """
    Compute the horizontal and vertical gradients using the Sobel operator.
    Then compute the gradient magnitude:
        M(x,y) = sqrt(Gx^2 + Gy^2)
    which acts as an edge confidence measure.
    Also compute the orientation:
        theta(x,y) = arctan2(Gy, Gx) wrapped to [0, 2*pi)
    """
    Gx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
    Gy = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
    magnitude = np.sqrt(Gx**2 + Gy**2)
    orientation = np.arctan2(Gy, Gx) % (2 * pi)
    return magnitude, orientation

# -------------------------------
# Step 2: Compute Block-Level DGC via Circular Variance
# -------------------------------
def block_dgc(mag_block, ori_block):
    """
    Compute the circular variance for a block.
    Circular variance (sigma) is computed as:
        sigma = 1 - (resultant / total_weight)
    where:
        resultant = sqrt((sum(M*cos(theta)))^2 + (sum(M*sin(theta)))^2)
        total_weight = sum(M)
    """
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    total_weight = np.sum(mag_block) + 1e-8  # avoid division by zero
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma

def compute_dgc(gray, block_size=8):
    """
    Divide the grayscale image into non-overlapping 8x8 blocks.
    For each block, compute the circular variance of edge orientations,
    then compute the global DGC as the average over all blocks.
    """
    magnitude, orientation = compute_gradients(gray)
    rows, cols = gray.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size
    block_values = []
    
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            r0, r1 = i * block_size, i * block_size + block_size
            c0, c1 = j * block_size, j * block_size + block_size
            mag_block = magnitude[r0:r1, c0:c1]
            ori_block = orientation[r0:r1, c0:c1]
            sigma = block_dgc(mag_block, ori_block)
            block_values.append(sigma)
    global_dgc = np.mean(block_values)
    return global_dgc

# -------------------------------
# Streamlit App Main Function
# -------------------------------
def main():
    st.title("Global Directional Gradient Consistency (DGC) Metric")
    st.write("""
        This app computes a simplified global DGC metric that quantifies the consistency of edge orientations 
        in an image. It uses the edge confidence (via Sobel gradient magnitude) and computes the block-wise 
        circular variance of edge orientations (with a hard-coded 8×8 block size). The global DGC is the average 
        over all blocks.
    """)
    
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Decode the image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error loading image. Please try a different file.")
            return
        
        st.image(img, caption="Uploaded Image", use_container_width=True)
        # Convert image to grayscale for processing
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Hard-coded block size for DGC computation (8x8)
        block_size = 8
        
        # Compute global DGC metric
        global_dgc = compute_dgc(gray, block_size)
        st.write(f"**Global DGC Metric:** {global_dgc:.4f}")

if __name__ == "__main__":
    main()
