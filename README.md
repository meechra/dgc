# PVD Stego‑Interference Detector with DGC Metric

Authors:
Mitra Vinda M. (21BCE5665), Mithun Balaji V (22BCE5189), Darshini S (22BCE1003)

School of Computer Science and Engineering (SCOPE), Vellore Institute of Technology Chennai

A Streamlit application that implements a Pixel Value Differencing (PVD) steganography detector using the Directional Gradient Coherence (DGC) metric and discrete wavelet analysis. By comparing the level of detail in raw and denoised wavelet detail images, the tool estimates the likelihood of hidden data. Click [here](https://dgc-stego.streamlit.app/) to access the app.

---

## Features

- **Wavelet Detail Extraction**: Applies a 2D DWT (Daubechies 1) to extract high‑frequency detail coefficients.
- **Bilateral Denoising**: Smooths the detail image while preserving edges for comparison.
- **Directional Gradient Coherence (DGC)**: Computes block‑based gradient coherence to measure local randomness.
- **Fusion Score**: Calculates the difference between raw and denoised DGC scores.
- **Likelihood Mapping**: Maps the fused score to a 0–100% likelihood of stego interference.
- **Visualizations**:
  - Original grayscale vs. denoised detail images
  - Difference map highlight
  - Number line indicating relative position between clean and stego medians

---

## Prerequisites

- Python 3.7+
- Streamlit
- OpenCV
- NumPy
- PyWavelets
- Matplotlib

Install dependencies with:

```bash
pip install streamlit opencv-python numpy PyWavelets matplotlib
```

---

## Usage

1. Clone this repository and navigate into its directory:
     ```bash
        git clone <repo-url>
        cd <repo-folder
     ```
2. Launch the Streamlit app:
   ```bash
      streamlit run app.py
   ```
3. In the browser, upload a grayscale image (PNG, JPG or JPEG).
4. Inspect the displayed metrics and visualizations:
   - **Raw DGC Score**: Coherence measure on wavelet detail.
   - **Denoised DGC Score**: Coherence after bilateral filtering.
   - **Difference (Fused Score)**: Raw – denoised.
   - **Likelihood of Stego Interference**: Percentage based on empirical medians.

---

## Configuration

The following parameters are hardcoded at the top of `app.py`:

| Parameter       | Description                                        | Default |
|-----------------|----------------------------------------------------|---------|
| `P_EXPONENT`    | Exponent in block DGC calculation                  | 2.5     |
| `WEIGHT_EXP`    | Power for block weight (sum of gradient magnitudes)| 2       |
| `GRAD_THRESHOLD`| Minimum block weight to include in DGC             | 1.0     |
| `BLOCK_SIZE`    | Size (pixels) of blocks for DGC computation        | 7       |
| `MEDIAN_CLEAN`  | Empirical median DGC score for clean images        | 0.0030  |
| `MEDIAN_STEGO`  | Empirical median DGC score for stego images        | 0.0018  |

These values were found to yield the most accurate results. They were derived from the experimentation phase of the app prototype, trained on the [PVD-Stego Dataset](https://www.kaggle.com/datasets/petrdufek/stego-pvd-dataset).

---

## How It Works

1. **Gradient Computation**:
   - Compute horizontal (`Gx`) and vertical (`Gy`) Sobel gradients.
   - Derive magnitude and orientation maps.

2. **Block‑based DGC**:
   - For each non‑overlapping block, compute coherence:
     1. Weight by gradient magnitudes.
     2. Measure angular dispersion via vector sum of orientations.
     3. Apply exponentiation to accentuate differences.

3. **Wavelet Analysis**:
   - Perform a single‑level 2D DWT to obtain detail subbands.
   - Normalize and convert to an 8‑bit image.
   - Denoise detail via bilateral filtering.

4. **Score Fusion & Mapping**:
   - Raw – denoised DGC scores yield the fused score.
   - Linearly map fused score relative to empirical clean and stego medians.
   - Clamp and convert to a percentage likelihood.

5. **Visualization**:
   - Display original vs. denoised detail, difference map, and relative position number line.

---

## License

This project is released under the MIT License. See [LICENSE](LICENSE) for details.

---




