import os
import sys

import numpy as np
import cv2
import streamlit as st
import matplotlib

matplotlib.use("Agg")  # we only show images in the page, no pop-up windows
import matplotlib.pyplot as plt

# Find the folder that contains FrequencyFilters.py (searching upwards from this file)
root = os.path.dirname(os.path.abspath(__file__))
while not os.path.exists(os.path.join(root, "FrequencyFilters.py")):
    parent = os.path.dirname(root)
    if parent == root:
        raise FileNotFoundError("FrequencyFilters.py was not found in any parent folder")
    root = parent
sys.path.insert(0, root)

import FrequencyFilters as ff  # our functions from the assignment script

OUT_DIR = os.path.join(root, "Assignment3_output")  # same output folder as the command-line script
KINDS = ["Arithmetic mean", "Median", "Max", "Min"]


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
    """Show an image inside a column. Color images are BGR (OpenCV order); gray ones have no channels."""
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
    results = {}  # filter name -> filtered image

    c1, c2 = st.columns(2)
    show(c1, img, "Original image")
    show(c2, noisy_sp, f"Salt & pepper noisy image, PSNR = {ff.psnr(img, noisy_sp):.2f} dB")

    cols = st.columns(4)
    for col, kind in zip(cols, KINDS):
        results[kind] = ff.window_filter(noisy_sp, kind, k)
        show(col, results[kind], f"{kind} filter, PSNR = {ff.psnr(img, results[kind]):.2f} dB")


# --------------------------------------------------------------------------
# Saving (same files and names as the command-line script)
# --------------------------------------------------------------------------
def build_figure_gaussian():
    """Combined figure of part 1 (same layout as exp2a_gaussian_adaptive_median.png)."""
    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle(f"Experiment 2a: Gaussian noise (σ = {sigma:g}) and adaptive median filter", fontsize=15)
    ff.show_img(ax[0], img, "Original image")
    ff.show_img(ax[1], noisy, f"Noisy image (before filter)\nPSNR = {ff.psnr(img, noisy):.2f} dB")
    ff.show_img(ax[2], restored, f"Adaptive median filter (after)\nPSNR = {ff.psnr(img, restored):.2f} dB")
    return fig


def build_figure_saltpepper():
    """Combined figure of part 2 (same layout as exp2b_saltpepper_filters.png)."""
    fig, ax = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    fig.suptitle(f"Experiment 2b: Salt & pepper noise ({100 * amount:g}%) and {k}x{k} filters", fontsize=15)
    ff.show_img(ax[0, 0], img, "Original image")
    ff.show_img(ax[0, 1], noisy_sp, f"Salt & pepper noisy image\nPSNR = {ff.psnr(img, noisy_sp):.2f} dB")
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), kind in zip(positions, KINDS):
        ff.show_img(ax[r, c], results[kind], f"{kind} filter\nPSNR = {ff.psnr(img, results[kind]):.2f} dB")
    return fig


st.sidebar.header("Save")

if st.sidebar.button("Save all images"):
    # every image in its own PNG file, inside Assignment3_output/exp2 (uses the current slider values)
    images = {
        "original": img,
        "gaussian_noisy": noisy,
        "gaussian_adaptive_median": restored,
        "saltpepper_noisy": noisy_sp,
    }
    for kind in KINDS:
        images["saltpepper_" + kind.lower().replace(" ", "_")] = results[kind]
    for name, im in images.items():
        ff.save_img(im, OUT_DIR, "exp2", name)
    st.sidebar.success(f"Saved {len(images)} images in {os.path.join(OUT_DIR, 'exp2')}")

if st.sidebar.button("Save combined figures"):
    # two PNGs (one per part of the experiment), inside Assignment3_output
    os.makedirs(OUT_DIR, exist_ok=True)
    for build, name in (
        (build_figure_gaussian, "exp2a_gaussian_adaptive_median.png"),
        (build_figure_saltpepper, "exp2b_saltpepper_filters.png"),
    ):
        fig = build()
        ff.save(fig, OUT_DIR, name)
        plt.close(fig)
    st.sidebar.success(f"Saved 2 figures in {OUT_DIR}")