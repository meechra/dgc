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
    Compute gradients using the Sobel operator:
      - Gx and Gy are the horizontal and vertical gradients.
      - Magnitude: M(x,y) = sqrt(Gx^2 + Gy^2) acts as the edge confidence measure.
      - Orientation: theta(x,y) = arctan2(Gy, Gx), wrapped to [0, 2*pi).
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
    Compute the circular variance (sigma) for a block.
    sigma = 1 - (resultant / total_weight)
    where:
      - resultant = sqrt((sum(M*cos(theta)))^2 + (sum(M*sin(theta)))^2)
      - total_weight = sum(M)
    A low sigma indicates that edge orientations are consistent.
    """
    sum_cos = np.sum(mag_block * np.cos(ori_block))
    sum_sin = np.sum(mag_block * np.sin(ori_block))
    total_weight = np.sum(mag_block) + 1e-8  # Avoid division by zero
    resultant = sqrt(sum_cos**2 + sum_sin**2)
    sigma = 1 - (resultant / total_weight)
    return sigma

def compute_dgc_map(gray, block_size=3):
    """
    Divide the grayscale image into non-overlapping blocks of size block_size x block_size.
    For each block, compute the local DGC (circular variance of edge orientations).
    Returns a 2D array (heatmap) of local DGC values.
    """
    magnitude, orientation = compute_gradients(gray)
    rows, cols = gray.shape
    num_blocks_row = rows // block_size
    num_blocks_col = cols // block_size
    dgc_map = np.zeros((num_blocks_row, num_blocks_col))
    
    for i in range(num_blocks_row):
        for j in range(num_blocks_col):
            r0, r1 = i * block_size, i * block_size + block_size
            c0, c1 = j * block_size, j * block_size + block_size
            mag_block = magnitude[r0:r1, c0:c1]
            ori_block = orientation[r0:r1, c0:c1]
            sigma = block_dgc(mag_block, ori_block)
            dgc_map[i, j] = sigma
    return dgc_map

def compute_global_dgc(gray, block_size=3):
    """
    Compute the global DGC metric as the average of the local DGC values.
    """
    dgc_map = compute_dgc_map(gray, block_size)
    global_dgc = np.mean(dgc_map)
    return global_dgc

# -------------------------------
# Step 3: Image Denoising (for Additive Gaussian Noise)
# -------------------------------
def denoise_image(gray):
    """
    Denoise the grayscale image using OpenCV's fast Non-Local Means Denoising.
    This method is effective for reducing additive Gaussian noise.
    """
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

# -------------------------------
# Streamlit App Main Function
# -------------------------------
def main():
    st.title("DGC Metric with Denoising Reference")
    st.write("""
        This app computes the local Directional Gradient Consistency (DGC) metric on 3×3 patches 
        for both a user-uploaded image and a denoised version of that image.
        
        It then displays:
        - The global DGC metric for the test (uploaded) image.
        - The global DGC metric for the denoised (cleaner) image.
        - The absolute difference between the global DGC metrics.
        - A heatmap of the absolute difference in local DGC values (per 3×3 patch).
    """)
    
    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Decode the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img is None:
            st.error("Error loading image. Please try a different file.")
            return
        
        st.image(img, caption="Uploaded Image", use_container_width=True)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Hard-coded block size for local DGC computation (3x3 patches)
        block_size = 3
        
        # Compute global DGC for the test image
        test_global_dgc = compute_global_dgc(gray, block_size)
        
        # Apply denoising to obtain a "cleaner" reference image
        denoised = denoise_image(gray)
        st.image(denoised, caption="Denoised Image", use_container_width=True)
        
        # Compute global DGC for the denoised image
        denoised_global_dgc = compute_global_dgc(denoised, block_size)
        
        # Compute absolute difference between the two global DGC metrics
        global_diff_direct = abs(test_global_dgc - denoised_global_dgc)
        
        # Compute local DGC maps for both images
        dgc_map_test = compute_dgc_map(gray, block_size)
        dgc_map_denoised = compute_dgc_map(denoised, block_size)
        
        # Compute the absolute difference map of local DGC values
        diff_map = np.abs(dgc_map_test - dgc_map_denoised)
        global_diff_local = np.mean(diff_map)
        
        st.write(f"**Global DGC (Test Image):** {test_global_dgc:.4f}")
        st.write(f"**Global DGC (Denoised Image):** {denoised_global_dgc:.4f}")
        st.write(f"**Absolute Global Difference (direct):** {global_diff_direct:.4f}")
        st.write(f"**Global Average Difference (local diff mean):** {global_diff_local:.4f}")
        
        # Plot the local DGC difference heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(diff_map, cmap='viridis', interpolation='nearest')
        ax.set_title("Local DGC Difference (3x3 Patches)")
        fig.colorbar(cax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
