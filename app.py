import streamlit as st
import cv2
import numpy as np
from math import sqrt, pi, exp
import matplotlib.pyplot as plt

# -------------------------------
# Step 1: Compute the Edge Confidence Map
# -------------------------------
def compute_gradients(gray):
    """
    Compute gradients using the Sobel operator:
      - Gx and Gy are the horizontal and vertical gradients.
      - Magnitude: M(x,y) = sqrt(Gx^2 + Gy^2) serves as the edge confidence measure.
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
    Compute the global DGC metric as the average of the local DGC values.
    """
    dgc_map = compute_dgc_map(gray, block_size)
    return np.mean(dgc_map)

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
# Step 4: Linear Normalization of the Difference (Without Clamping)
# -------------------------------
def normalize_diff(raw_diff, d_min=-0.1611, d_max=0.0575):
    """
    Normalize the raw difference between global DGC scores to the range [0,1] using linear min-max scaling.
    The transformation is:
    
        normalized = 1 - ((raw_diff - d_min) / (d_max - d_min))
    
    This maps:
      - A raw difference equal to d_min (-0.1611) to 1,
      - A raw difference equal to d_max (0.0575) to 0.
    
    Unlike hard clamping, we return the continuous value so that the intensity of interference is preserved.
    """
    normalized = 1.0 - ((raw_diff - d_min) / (d_max - d_min))
    return normalized

# -------------------------------
# Streamlit App Main Function
# -------------------------------
def main():
    st.title("DGC Metric with Grayscale Denoising & Linear Difference Normalization")
    st.write("""
        This app computes the global Directional Gradient Consistency (DGC) metric on patches for a user-uploaded image 
        and its denoised version (using grayscale denoising). The image is processed in grayscale and divided into 
        non-overlapping patches (the patch size is tunable via a slider, from 3×3 up to 8×8). The global DGC is 
        the average of the local DGC (circular variance of edge orientations) over all patches.
        
        The raw difference between the global DGC scores of the test image and its denoised version is then normalized 
        using linear min-max scaling (without hard clamping) to the range [0,1]. In this normalized scale, a clean image 
        should have a value closer to 0, while a stego-interfered image should have a value closer to 1.
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
        
        # Create a slider for patch (block) size selection (from 3x3 to 8x8)
        block_size = st.slider("Select Patch (Block) Size", min_value=3, max_value=8, value=5, step=1,
                               help="Local DGC will be computed on non-overlapping patches of size block_size x block_size")
        
        # Compute global DGC scores for the test and denoised images
        test_global_dgc = compute_global_dgc(gray_test, block_size)
        denoised_global_dgc = compute_global_dgc(gray_denoised, block_size)
        
        # Compute the raw difference between the two global DGC scores
        raw_diff = test_global_dgc - denoised_global_dgc
        
        # Normalize the difference using linear min-max scaling (without hard clamping)
        norm_diff = normalize_diff(raw_diff)
        
        # Also compute local DGC maps for both images and then the absolute difference map
        dgc_map_test = compute_dgc_map(gray_test, block_size)
        dgc_map_denoised = compute_dgc_map(gray_denoised, block_size)
        diff_map = np.abs(dgc_map_test - dgc_map_denoised)
        global_diff_local = np.mean(diff_map)
        
        st.write(f"**Global DGC (Test Image):** {test_global_dgc:.4f}")
        st.write(f"**Global DGC (Denoised Image):** {denoised_global_dgc:.4f}")
        st.write(f"**Raw Global Difference (Test - Denoised):** {raw_diff:.4f}")
        st.write(f"**Normalized Global Difference:** {norm_diff:.4f}")
        st.write(f"**Global Average Local Difference (patch mean):** {global_diff_local:.4f}")
        
        # Plot the local DGC difference heatmap
        fig, ax = plt.subplots()
        cax = ax.imshow(diff_map, cmap='viridis', interpolation='nearest')
        ax.set_title(f"Local DGC Difference ({block_size}x{block_size} Patches)")
        fig.colorbar(cax)
        st.pyplot(fig)

if __name__ == "__main__":
    main()
