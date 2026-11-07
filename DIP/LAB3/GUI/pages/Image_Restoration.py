import os
import sys

import numpy as np
import cv2
import streamlit as st
import matplotlib

matplotlib.use("Agg")  # we only show images in the page, no pop-up windows

# Find the folder that contains FrequencyFilters.py (searching upwards from this file)
root = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(root, "FrequencyFilters.py")):
    parent = os.path.dirname(root)
    if parent == root:
        raise FileNotFoundError("FrequencyFilters.py was not found in any parent folder")
    root = parent
sys.path.insert(0, root)

import FrequencyFilters as ff  # our functions from the assignment script


# --------------------------------------------------------------------------
# Small helpers
# --------------------------------------------------------------------------
def get_image(uploaded, stems):
    """Uploaded file if there is one, else the default image next to FrequencyFilters.py."""
    if uploaded is not None:
        data = np.frombuffer(uploaded.read(), np.uint8)
        return cv2.imdecode(data, cv2.IMREAD_COLOR)
    path = ff.find_image(root, stems)
    return ff.load_color(path) if path else None


def show(col, img, caption):
    """Show an image inside a column. Color images are BGR (OpenCV order); gray ones (masks, spectra) have no channels."""
    if img.ndim == 3:
        col.image(img, caption=caption, channels="BGR")
    else:
        col.image(img, caption=caption)


# --------------------------------------------------------------------------
# Page
# --------------------------------------------------------------------------
st.set_page_config(page_title="Image Restoration", layout="wide")
st.title("Experiment 2: Image Restoration (Lenna)")

st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload an image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)

img = get_image(uploaded, ["lenna", "Lenna"])
if img is None:
    st.warning("No image found. Upload one from the sidebar.")
    st.stop()

tab1, tab2 = st.tabs(["Gaussian noise + Adaptive median", "Salt & pepper noise + 4 filters"])

# ---- Part 1: Gaussian noise + adaptive median filter ----
with tab1:
    sigma = st.slider("Noise sigma", 1, 100, 20)
    s_max = st.select_slider("Largest window of the adaptive median (S_max)", options=[3, 5, 7, 9, 11], value=7)

    noisy = ff.add_gaussian_noise(img, sigma)
    restored = ff.adaptive_median(noisy, s_max)

    c1, c2, c3 = st.columns(3)
    show(c1, img, "Original image")
    show(c2, noisy, f"Noisy image (before filter), PSNR = {ff.psnr(img, noisy):.2f} dB")
    show(c3, restored, f"Adaptive median (after), PSNR = {ff.psnr(img, restored):.2f} dB")

# ---- Part 2: salt & pepper noise + mean / median / max / min ----
with tab2:
    amount = st.slider("Noise amount (% of values)", 1, 50, 5) / 100
    k = st.select_slider("Filter window size", options=[3, 5, 7, 9], value=3)

    noisy_sp = ff.add_salt_pepper(img, amount)

    c1, c2 = st.columns(2)
    show(c1, img, "Original image")
    show(c2, noisy_sp, f"Salt & pepper noisy image, PSNR = {ff.psnr(img, noisy_sp):.2f} dB")

    cols = st.columns(4)
    for col, kind in zip(cols, ["Arithmetic mean", "Median", "Max", "Min"]):
        out = ff.window_filter(noisy_sp, kind, k)
        show(col, out, f"{kind} filter, PSNR = {ff.psnr(img, out):.2f} dB")