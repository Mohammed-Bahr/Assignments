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
st.set_page_config(page_title="Fourier Transform", layout="wide")
st.title("Experiment 1: Fourier Transform (cameraman)")

# ---- controls (sidebar) ----
st.sidebar.header("Controls")
uploaded = st.sidebar.file_uploader(
    "Upload an image (optional)", type=["png", "jpg", "jpeg", "tif", "tiff", "bmp"]
)
kind = st.sidebar.radio("Filter shape", ["ideal", "gaussian"])
cutoff = st.sidebar.slider("Cutoff D0 (pixels)", 1, 100, 30)

img = get_image(uploaded, ["cameraman", "Cameraman"])
if img is None:
    st.warning("No image found. Upload one from the sidebar.")
    st.stop()

# ---- computation (same steps as experiment_1 in FrequencyFilters.py) ----
F = ff.to_frequency(img)  # 1. DFT
spec_orig, ref_max = ff.spectrum_to_u8(F)

H_low = ff.lowpass_mask(img.shape, cutoff, kind)  # 2. low-pass filter
H_high = ff.highpass_mask(img.shape, cutoff, kind)  # 3. high-pass filter
F_low = F * H_low
F_high = F * H_high
spec_low, _ = ff.spectrum_to_u8(F_low, ref_max)
spec_high, _ = ff.spectrum_to_u8(F_high, ref_max)

low_raw = ff.to_spatial(F_low)  # 4. back to the spatial domain
high_raw = ff.to_spatial(F_high)
low_img = ff.to_u8(low_raw)
high_img = ff.scale_abs_to_u8(high_raw)

mask_low = ff.to_u8(H_low[..., 0] * 255)
mask_high = ff.to_u8(H_high[..., 0] * 255)

energy = float(np.sum(np.abs(F) ** 2))
energy_low = 100 * float(np.sum(np.abs(F_low) ** 2)) / energy
energy_high = 100 * float(np.sum(np.abs(F_high) ** 2)) / energy

# ---- numbers ----
m1, m2, m3 = st.columns(3)
m1.metric("Energy kept by LPF", f"{energy_low:.2f} %")
m2.metric("Energy kept by HPF", f"{energy_high:.2f} %")
m3.metric("LPF result PSNR", f"{ff.psnr(img, low_img):.2f} dB")

# ---- images ----
st.subheader("Original")
c1, c2, c3 = st.columns(3)
show(c1, img, "Original image")
show(c2, spec_orig, "Original spectrum (log magnitude)")

st.subheader("Low-pass filter (LPF)")
c1, c2, c3 = st.columns(3)
show(c1, mask_low, "LPF mask")
show(c2, spec_low, "Spectrum after LPF")
show(c3, low_img, "Spatial domain after LPF")

st.subheader("High-pass filter (HPF)")
c1, c2, c3 = st.columns(3)
show(c1, mask_high, "HPF mask")
show(c2, spec_high, "Spectrum after HPF")
show(c3, high_img, "Spatial domain after HPF (|result|, scaled)")


# --------------------------------------------------------------------------
# Saving (same files and names as the command-line script)
# --------------------------------------------------------------------------
def build_figure():
    """The combined figure of the experiment (same layout as exp1_frequency_filtering.png)."""
    fig, ax = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(f"Experiment 1: {kind.capitalize()} low-pass / high-pass filtering (D0 = {cutoff:g})", fontsize=15)
    ff.show_img(ax[0, 0], img, "Original image")
    ff.show_img(ax[0, 1], spec_orig, "Original spectrum (log magnitude)")
    ax[0, 2].axis("off")
    ff.show_img(ax[1, 0], mask_low, "Low-pass filter (LPF)")
    ff.show_img(ax[1, 1], spec_low, "Spectrum after LPF")
    ff.show_img(ax[1, 2], low_img, "Spatial domain after LPF")
    ff.show_img(ax[2, 0], mask_high, "High-pass filter (HPF)")
    ff.show_img(ax[2, 1], spec_high, "Spectrum after HPF")
    ff.show_img(ax[2, 2], high_img, "Spatial domain after HPF (|result|, scaled)")
    return fig


st.sidebar.header("Save")

if st.sidebar.button("Save all images"):
    # every image in its own PNG file, inside Assignment3_output/exp1
    images = {
        "original": img,
        "spectrum_original": spec_orig,
        "lpf_mask": mask_low,
        "hpf_mask": mask_high,
        "spectrum_after_lpf": spec_low,
        "spectrum_after_hpf": spec_high,
        "lpf_result": low_img,
        "hpf_result": high_img,
    }
    for name, im in images.items():
        ff.save_img(im, OUT_DIR, "exp1", name)
    st.sidebar.success(f"Saved {len(images)} images in {os.path.join(OUT_DIR, 'exp1')}")

if st.sidebar.button("Save combined figure"):
    # one PNG with the whole experiment, inside Assignment3_output
    os.makedirs(OUT_DIR, exist_ok=True)
    fig = build_figure()
    ff.save(fig, OUT_DIR, "exp1_frequency_filtering.png")
    plt.close(fig)
    st.sidebar.success(f"Saved {os.path.join(OUT_DIR, 'exp1_frequency_filtering.png')}")