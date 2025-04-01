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
      - Orientation: theta(x,y) = arctan2(Gy, Gx) wrapped to [0, 2*pi).
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

def compute_dgc_map(gray, block_size):
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

def compute_global_dgc(gray, block_size):
    """
    Compute the global DGC metric as the average of the local DGC values using non-overlapping blocks.
    """
    dgc_map = compute_dgc_map(gray, block_size)
    return np.mean(dgc_map)

# -------------------------------
# New Functions: Overlapping and Composite DGC
# -------------------------------
def compute_dgc_map_overlapping(gray, block_size, step):
    """
    Compute a DGC map using overlapping blocks.
    The window slides over the image with a specified step size.
    Returns a list of local DGC values computed at each window position.
    """
    magnitude, orientation = compute_gradients(gray)
    rows, cols = gray.shape
    dgc_values = []
    positions = []
    for i in range(0, rows - block_size + 1, step):
        for j in range(0, cols - block_size + 1, step):
            mag_block = magnitude[i:i+block_size, j:j+block_size]
            ori_block = orientation[i:i+block_size, j:j+block_size]
            sigma = block_dgc(mag_block, ori_block)
            dgc_values.append(sigma)
            positions.append((i, j))
    return np.array(dgc_values), positions

def compute_global_dgc_overlapping(gray, block_size, step):
    """
    Compute the global DGC metric as the average of the local DGC values using overlapping blocks.
    """
    dgc_vals, _ = compute_dgc_map_overlapping(gray, block_size, step)
    return np.mean(dgc_vals)

def compute_composite_dgc(gray, scales, overlap_ratio=0.5):
    """
    Compute a composite DGC metric across multiple scales using overlapping blocks.
    For each scale (block size) in scales, the function computes a global DGC metric using
    overlapping blocks (with step determined by the overlap_ratio). The composite metric is the average
    of the global DGC values from all scales.
    """
    composite_values = []
    for block_size in scales:
        # Determine step size: ensure at least one pixel shift
        step = max(1, int(block_size * (1 - overlap_ratio)))
        global_dgc = compute_global_dgc_overlapping(gray, block_size, step)
        composite_values.append(global_dgc)
    return np.mean(composite_values)

# -------------------------------
# Step 3: Grayscale Image Denoising
# -------------------------------
def denoise_gray(gray):
    """
    Denoise the grayscale image using OpenCV's fast Non-Local Means Denoising for grayscale images.
    This method effectively reduces additive Gaussian noise while preserving edges.
    """
    denoised = cv2.fastNlMeansDenoising(gray, None, h=10, templateWindowSize=7, searchWindowSize=21)
    return denoised

# -------------------------------
# Streamlit App Main Function
# -------------------------------
def main():
    st.title("DGC Metric with Grayscale Denoising and Composite Multi-Scale Computation")
    st.write("""
        This app computes the local Directional Gradient Consistency (DGC) metric on patches for a user-uploaded image 
        and its denoised version (using grayscale denoising). 
        
        The image is converted to grayscale and edge gradients are computed using the Sobel operator. The image can be analyzed using:
        
        - **Non-overlapping blocks:** The image is divided into non-overlapping patches (user-selectable size).
        - **Composite multi-scale DGC (overlapping blocks):** The image is analyzed at multiple scales using overlapping blocks.
        
        For the composite metric, overlapping windows are used at several block sizes and the resulting global DGC values are averaged.
    """)

    uploaded_file = st.file_uploader("Upload an image (jpg, jpeg, png)", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Read the uploaded image using OpenCV
        file_bytes = np.asarray(bytearray(uploaded_file.read()), dtype=np.uint8)
        img_color = cv2.imdecode(file_bytes, cv2.IMREAD_COLOR)
        if img_color is None:
            st.error("Error loading image. Please try a different file.")
            return
        
        st.image(img_color, caption="Uploaded Image", use_container_width=True)
        
        # Convert the color image to grayscale for processing
        gray_test = cv2.cvtColor(img_color, cv2.COLOR_BGR2GRAY)
        
        # Apply grayscale denoising
        gray_denoised = denoise_gray(gray_test)
        st.image(gray_denoised, caption="Denoised Grayscale Image", use_container_width=True)
        
        # Option to select computation method
        composite_mode = st.checkbox("Compute Composite Multi-Scale DGC (using overlapping blocks)")
        
        if composite_mode:
            st.subheader("Composite Multi-Scale DGC Computation")
            # Define a set of scales (block sizes) to analyze
            scales = st.multiselect("Select Block Sizes (scales) for composite computation", 
                                      options=[3,4,5,6,7,8], default=[3,5,7])
            overlap_ratio = st.slider("Select Overlap Ratio", min_value=0.0, max_value=0.9, value=0.5, step=0.1,
                                       help="Fraction of overlap between adjacent blocks (0.0 = no overlap, 0.9 = high overlap)")
            if not scales:
                st.error("Please select at least one block size.")
                return
            
            # Compute composite global DGC for the test and denoised images
            test_composite_dgc = compute_composite_dgc(gray_test, scales, overlap_ratio)
            denoised_composite_dgc = compute_composite_dgc(gray_denoised, scales, overlap_ratio)
            
            composite_diff = abs(test_composite_dgc - denoised_composite_dgc)
            
            st.write(f"**Composite Global DGC (Test Image):** {test_composite_dgc:.4f}")
            st.write(f"**Composite Global DGC (Denoised Image):** {denoised_composite_dgc:.4f}")
            st.write(f"**Composite Global Absolute Difference:** {composite_diff:.4f}")
        else:
            st.subheader("Standard Non-Overlapping DGC Computation")
            block_size = st.slider("Select Patch (Block) Size", min_value=3, max_value=8, value=5, step=1,
                                   help="The patch size for local DGC computation (patch will be block_size x block_size)")
            # Compute global DGC for the test image
            test_global_dgc = compute_global_dgc(gray_test, block_size)
            # Compute global DGC for the denoised image
            denoised_global_dgc = compute_global_dgc(gray_denoised, block_size)
            global_diff_direct = abs(test_global_dgc - denoised_global_dgc)
            
            # Compute local DGC maps for both images and then the absolute difference map
            dgc_map_test = compute_dgc_map(gray_test, block_size)
            dgc_map_denoised = compute_dgc_map(gray_denoised, block_size)
            diff_map = np.abs(dgc_map_test - dgc_map_denoised)
            global_diff_local = np.mean(diff_map)
            
            st.write(f"**Global DGC (Test Image):** {test_global_dgc:.4f}")
            st.write(f"**Global DGC (Denoised Image):** {denoised_global_dgc:.4f}")
            st.write(f"**Global Absolute Difference (direct):** {global_diff_direct:.4f}")
            st.write(f"**Global Average Difference (local diff mean):** {global_diff_local:.4f}")
            
            # Plot the local DGC difference heatmap
            fig, ax = plt.subplots()
            cax = ax.imshow(diff_map, cmap='viridis', interpolation='nearest')
            ax.set_title(f"Local DGC Difference ({block_size}x{block_size} Patches)")
            fig.colorbar(cax)
            st.pyplot(fig)

if __name__ == "__main__":
    main()
