#!/usr/bin/env python3
"""
Digital Image Processing - Assignment 3
Frequency-domain filtering & noise removal (RGB / color version)

Experiment 1 (cameraman): DFT -> low-pass / high-pass filter in the frequency domain -> inverse DFT
Experiment 2 (Lenna):
    a) Gaussian noise  -> adaptive median filter
    b) Salt & pepper noise -> arithmetic mean / median / max / min filters

Color handling: every operation is applied independently to each of the B, G, R channels
(np.fft and the OpenCV filters do this natively on (M, N, 3) arrays, and the noise is generated
independently per channel). Images are kept in OpenCV's BGR order internally and converted
to RGB only for display with matplotlib.

Usage:
    python FrequencyFilters_qtGUI.py                                 # looks for cameraman / lenna next to the script, else synthetic images
    python FrequencyFilters_qtGUI.py --cameraman cameraman.png --lenna lenna.png
    python FrequencyFilters_qtGUI.py --cutoff 40 --filter gaussian   # cutoff radius (pixels) and filter shape
    python FrequencyFilters_qtGUI.py --sigma 30 --sp 0.1 --ksize 5   # noise strength / filter window
    python FrequencyFilters_qtGUI.py --no-show                       # only save figures (no pop-up windows)
    python FrequencyFilters_qtGUI.py --gui                           # open the PyQt5 desktop GUI

Figures and result images are saved to ./Assignment3_output (change with --out).
"""

import argparse
import os
import sys

import numpy as np
import cv2
import matplotlib

# The backend must be chosen before pyplot is imported.
if "--no-show" in sys.argv:
    matplotlib.use("Agg")
import matplotlib.pyplot as plt


# --------------------------------------------------------------------------
# Helpers
# --------------------------------------------------------------------------
def load_color(path):
    """Load an image as 8-bit color (shape (M, N, 3), dtype uint8, channel order BGR)."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        sys.exit(
            f"Error: could not read image '{path}'. Check the path and file format."
        )
    return img


def find_image(script_dir, stems):
    """Look next to the script for <stem>.<ext> for every name/extension combination."""
    exts = [".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"]
    for stem in stems:
        for ext in exts:
            p = os.path.join(script_dir, stem + ext)
            if os.path.exists(p):
                return p
    return None


def get_image(path, stems, script_dir, demo_fn, out_dir):
    """Use the given path, else a file named stems[i] next to the script, else a synthetic demo image."""
    if path:
        return load_color(path)
    found = find_image(script_dir, stems)
    if found:
        print(f"  using '{found}'")
        return load_color(found)

    print(f"  '{stems[0]}' not found -> using a synthetic demo image.")
    img = demo_fn()
    cv2.imwrite(os.path.join(out_dir, f"demo_{stems[0]}.png"), img)
    return img


def colorize(gray, gains_bgr):
    """Turn a gray demo image into a BGR image by scaling each channel (only used for demo images)."""
    img = gray[..., None].astype(np.float64) * np.array(gains_bgr, dtype=np.float64)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_cameraman(h=256, w=256, seed=0):
    """Synthetic gray 'cameraman-like' scene (stand-in for cameraman): sky, ground, figure, texture."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = np.where(y < h * 0.65, 170 - 40 * (y / h), 90.0)  # sky gradient / ground
    img = img + 12 * np.sin(x / 3.0) * (y >= h * 0.65)  # grass-like texture on the ground
    body = (abs(x - w * 0.4) < w * 0.10) & (y > h * 0.35) & (y < h * 0.85)
    head = ((x - w * 0.4) ** 2 + (y - h * 0.28) ** 2) < (h * 0.07) ** 2
    tripod = (abs(x - w * 0.72) < 3) & (y > h * 0.4)
    img = np.where(body, 30, img)
    img = np.where(head, 60, img)
    img = np.where(tripod, 20, img)
    img = img + rng.normal(0, 2, (h, w))
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)


def make_demo_scene(h=256, w=256, seed=0):
    """Synthetic color image with smooth areas, sharp edges and some texture (stand-in for Lenna)."""
    rng = np.random.default_rng(seed)
    y, x = np.mgrid[0:h, 0:w]
    img = 90 + 60 * (x / w)  # soft background gradient
    disk = ((x - w * 0.3) ** 2 + (y - h * 0.5) ** 2) < (h * 0.28) ** 2
    rect = (abs(x - w * 0.72) < w * 0.16) & (abs(y - h * 0.5) < h * 0.25)
    img = img + 80 * disk - 60 * rect
    img = img + 10 * np.sin(x / 4.0) * rect  # texture inside the rectangle
    img = img + rng.normal(0, 1.5, (h, w))
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return colorize(gray, (0.7, 1.0, 0.8))  # greenish tint (B, G, R gains)


def add_gaussian_noise(img, sigma, mean=0.0, seed=0):
    """Add Gaussian noise N(mean, sigma^2) and clip back to [0, 255].
    img.shape includes the channel axis, so every channel gets its own independent noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8)


def add_salt_pepper(img, amount, seed=0):
    """Salt & pepper noise: a fraction `amount` of the values become 0 (pepper) or 255 (salt), half each.
    Drawn independently per channel (img.shape includes the channel axis)."""
    rng = np.random.default_rng(seed)
    r = rng.random(img.shape)
    noisy = img.copy()
    noisy[r < amount / 2] = 0  # pepper
    noisy[(r >= amount / 2) & (r < amount)] = 255  # salt
    return noisy


def mse(a, b):
    """Mean squared error between two images (averaged over all pixels and channels)."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    """Peak signal-to-noise ratio in dB (higher = closer to the reference)."""
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0**2 / m)

# --------------------------------------------------------------------------
# Frequency domain (all of it acts channel-by-channel on (M, N, 3) images)
# --------------------------------------------------------------------------
def to_frequency(img):
    """2-D DFT of every channel, with the zero frequency shifted to the centre of the array."""
    F = np.fft.fft2(img.astype(np.float64), axes=(0, 1))
    return np.fft.fftshift(F, axes=(0, 1))


def to_spatial(F):
    """Inverse of to_frequency: un-shift, inverse DFT, keep the real part (the imaginary part is ~0)."""
    f = np.fft.ifft2(np.fft.ifftshift(F, axes=(0, 1)), axes=(0, 1))
    return np.real(f)


def distance_grid(shape):
    """D(u, v) = distance of every frequency sample from the centre of the shifted spectrum."""
    M, N = shape[:2]
    u = np.arange(M) - M // 2
    v = np.arange(N) - N // 2
    V, U = np.meshgrid(v, u)
    return np.hypot(U, V)


def lowpass_mask(shape, cutoff, kind):
    """Low-pass filter H(u, v), shape (M, N, 1) so it broadcasts over the channels.
    ideal   : H = 1 if D <= cutoff else 0
    gaussian: H = exp(-D^2 / (2 * cutoff^2))"""
    D = distance_grid(shape)
    if kind == "ideal":
        H = (D <= cutoff).astype(np.float64)
    elif kind == "gaussian":
        H = np.exp(-(D**2) / (2.0 * cutoff**2))
    else:
        raise ValueError(kind)
    return H[..., None]


def highpass_mask(shape, cutoff, kind):
    """High-pass filter = 1 - low-pass filter (same cutoff and shape)."""
    return 1.0 - lowpass_mask(shape, cutoff, kind)


def spectrum_to_u8(F, ref_max=None):
    """Display a spectrum: log(1 + |F|) averaged over the channels, scaled to 0..255.
    Pass the same ref_max for several spectra to display them on a common scale."""
    s = np.log1p(np.abs(F)).mean(axis=2)
    if ref_max is None:
        ref_max = float(s.max())
    return np.clip(s * (255.0 / ref_max), 0, 255).astype(np.uint8), ref_max


def to_u8(x):
    """Round and clip a float image to [0, 255] as uint8."""
    return np.clip(np.round(x), 0, 255).astype(np.uint8)


def scale_abs_to_u8(x):
    """|x| scaled for display without letting a few strong values dominate (99th percentile -> 255).
    One common scale is used for all channels so colors are preserved. Used for the high-pass result,
    which is not a normal image: it has no DC term, so it is ~0 on flat areas and +/- on edges."""
    magnitude = np.abs(x).astype(np.float64)
    limit = float(np.percentile(magnitude, 99))
    if limit <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(magnitude * (255.0 / limit), 0, 255).astype(np.uint8)


# --------------------------------------------------------------------------
# Spatial noise-removal filters
# --------------------------------------------------------------------------
def window_filter(img, kind, k):
    """k x k window filter: 'Arithmetic mean', 'Median', 'Max' or 'Min'."""
    kernel = np.ones((k, k), np.uint8)
    if kind == "Arithmetic mean":
        return cv2.blur(img, (k, k))
    if kind == "Median":
        return cv2.medianBlur(img, k)
    if kind == "Max":
        return cv2.dilate(img, kernel)  # dilation with a flat window = local maximum
    if kind == "Min":
        return cv2.erode(img, kernel)  # erosion with a flat window = local minimum
    raise ValueError(kind)


def adaptive_median(img, s_max=7):
    """Adaptive median filter (textbook algorithm), vectorized over the whole image.
    For every pixel the window grows 3x3, 5x5, ... up to s_max:
      Stage A: zmin < zmed < zmax ?  yes -> go to stage B,  no -> grow the window
               (if the window already is s_max the output is zmed)
      Stage B: zmin < z_xy < zmax ?  yes -> output z_xy (keep the pixel),  no -> output zmed
    zmin / zmax / zmed are the local min / max / median (erode / dilate / medianBlur)."""
    result = img.copy()
    done = np.zeros(img.shape, dtype=bool)  # pixels already decided at a smaller window
    zmed = img
    for w in range(3, s_max + 1, 2):
        kernel = np.ones((w, w), np.uint8)
        zmin = cv2.erode(img, kernel)
        zmax = cv2.dilate(img, kernel)
        zmed = cv2.medianBlur(img, w)
        stage_a = (zmed > zmin) & (zmed < zmax)
        stage_b = (img > zmin) & (img < zmax)
        newly = stage_a & ~done
        result[newly] = np.where(stage_b, img, zmed)[newly]
        done |= stage_a
    result[~done] = zmed[~done]  # window reached s_max without passing stage A
    return result


# --------------------------------------------------------------------------
# Plot helpers
# --------------------------------------------------------------------------
def show_img(ax, img, title):
    if img.ndim == 3:
        # OpenCV stores BGR, matplotlib expects RGB
        ax.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    else:
        ax.imshow(
            img, cmap="gray", vmin=0, vmax=255
        )  # fixed range so brightness is comparable
    ax.set_title(title, fontsize=10)
    ax.axis("off")


def save(fig, out_dir, name):
    path = os.path.join(out_dir, name)
    fig.savefig(path, dpi=160, bbox_inches="tight")
    print(f"  saved {path}")


def save_img(img, out_dir, sub, name):
    """Save a single image (lossless PNG) in out_dir/sub/name.png. (BGR is what cv2.imwrite expects.)"""
    folder = os.path.join(out_dir, sub)
    os.makedirs(folder, exist_ok=True)
    cv2.imwrite(os.path.join(folder, name + ".png"), img)


# --------------------------------------------------------------------------
# Experiments
# --------------------------------------------------------------------------
def experiment_1(img, cutoff, kind, out_dir):
    print("\n=== Experiment 1: frequency-domain filtering (cameraman) ===")
    print(f"  image size: {img.shape[1]}x{img.shape[0]}   filter: {kind}   cutoff D0 = {cutoff:g} px")

    # 1. DFT: image -> frequency domain
    F = to_frequency(img)
    spec_orig, ref_max = spectrum_to_u8(F)

    # 2 + 3. low-pass / high-pass filters and their product with the spectrum
    H_low = lowpass_mask(img.shape, cutoff, kind)
    H_high = highpass_mask(img.shape, cutoff, kind)
    F_low = F * H_low
    F_high = F * H_high
    spec_low, _ = spectrum_to_u8(F_low, ref_max)
    spec_high, _ = spectrum_to_u8(F_high, ref_max)

    # 4. back to the spatial domain
    low_raw = to_spatial(F_low)
    high_raw = to_spatial(F_high)
    low_img = to_u8(low_raw)
    high_img = scale_abs_to_u8(high_raw)  # |high-pass| scaled for display

    energy = float(np.sum(np.abs(F) ** 2))
    print(f"  energy kept by LPF: {100 * np.sum(np.abs(F_low) ** 2) / energy:.2f}%   "
          f"by HPF: {100 * np.sum(np.abs(F_high) ** 2) / energy:.2f}%")
    print(f"  original   : mean={img.mean():.1f}  std={img.std():.1f}")
    print(f"  LPF result : mean={low_raw.mean():.1f}  std={low_raw.std():.1f}   PSNR vs original = {psnr(img, low_img):.2f} dB")
    print(f"  HPF result : mean={high_raw.mean():.1f}  std={high_raw.std():.1f}   (no DC term -> mean ~ 0, values are +/-)")

    mask_low = to_u8(H_low[..., 0] * 255)
    mask_high = to_u8(H_high[..., 0] * 255)

    save_img(img, out_dir, "exp1", "original")
    save_img(spec_orig, out_dir, "exp1", "spectrum_original")
    save_img(mask_low, out_dir, "exp1", "lpf_mask")
    save_img(mask_high, out_dir, "exp1", "hpf_mask")
    save_img(spec_low, out_dir, "exp1", "spectrum_after_lpf")
    save_img(spec_high, out_dir, "exp1", "spectrum_after_hpf")
    save_img(low_img, out_dir, "exp1", "lpf_result")
    save_img(high_img, out_dir, "exp1", "hpf_result")

    # single grid: columns = filter / frequency domain / spatial domain
    fig, ax = plt.subplots(3, 3, figsize=(13, 12), constrained_layout=True)
    fig.suptitle(f"Experiment 1: {kind.capitalize()} low-pass / high-pass filtering (D0 = {cutoff:g})", fontsize=15)
    show_img(ax[0, 0], img, "Original image")
    show_img(ax[0, 1], spec_orig, "Original spectrum (log magnitude)")
    ax[0, 2].axis("off")
    show_img(ax[1, 0], mask_low, "Low-pass filter (LPF)")
    show_img(ax[1, 1], spec_low, "Spectrum after LPF")
    show_img(ax[1, 2], low_img, "Spatial domain after LPF")
    show_img(ax[2, 0], mask_high, "High-pass filter (HPF)")
    show_img(ax[2, 1], spec_high, "Spectrum after HPF")
    show_img(ax[2, 2], high_img, "Spatial domain after HPF (|result|, scaled)")
    save(fig, out_dir, "exp1_frequency_filtering.png")


def experiment_2(img, sigma, sp_amount, k, s_max, out_dir):
    print("\n=== Experiment 2: noise removal (Lenna) ===")
    print(f"  image size: {img.shape[1]}x{img.shape[0]}")
    save_img(img, out_dir, "exp2", "original")

    # ---- 2.1 Gaussian noise + adaptive median filter ----
    print(f"\n  -- Gaussian noise (mean=0, sigma={sigma:g}) + adaptive median filter (S_max={s_max}) --")
    noisy_g = add_gaussian_noise(img, sigma)
    adapt = adaptive_median(noisy_g, s_max)
    print(f"  noisy image    : PSNR = {psnr(img, noisy_g):.2f} dB")
    print(f"  adaptive median: PSNR = {psnr(img, adapt):.2f} dB")
    save_img(noisy_g, out_dir, "exp2", "gaussian_noisy")
    save_img(adapt, out_dir, "exp2", "gaussian_adaptive_median")

    fig, ax = plt.subplots(1, 3, figsize=(15, 5.5), constrained_layout=True)
    fig.suptitle(f"Experiment 2a: Gaussian noise (σ = {sigma:g}) and adaptive median filter", fontsize=15)
    show_img(ax[0], img, "Original image")
    show_img(ax[1], noisy_g, f"Noisy image (before filter)\nPSNR = {psnr(img, noisy_g):.2f} dB")
    show_img(ax[2], adapt, f"Adaptive median filter (after)\nPSNR = {psnr(img, adapt):.2f} dB")
    save(fig, out_dir, "exp2a_gaussian_adaptive_median.png")

    # ---- 2.2 Salt & pepper noise + four filters ----
    print(f"\n  -- Salt & pepper noise ({100 * sp_amount:g}% of values) + {k}x{k} filters --")
    noisy_sp = add_salt_pepper(img, sp_amount)
    print(f"  noisy image    : PSNR = {psnr(img, noisy_sp):.2f} dB")
    save_img(noisy_sp, out_dir, "exp2", "saltpepper_noisy")

    kinds = ["Arithmetic mean", "Median", "Max", "Min"]
    results = {}  # kind -> (image, psnr)
    for kind in kinds:
        out = window_filter(noisy_sp, kind, k)
        results[kind] = (out, psnr(img, out))
        print(f"  {kind:<15}: PSNR = {results[kind][1]:.2f} dB")
        save_img(out, out_dir, "exp2", "saltpepper_" + kind.lower().replace(" ", "_"))

    # single grid: original / noisy / mean  +  median / max / min
    fig, ax = plt.subplots(2, 3, figsize=(15, 10.5), constrained_layout=True)
    fig.suptitle(f"Experiment 2b: Salt & pepper noise ({100 * sp_amount:g}%) and {k}x{k} filters", fontsize=15)
    show_img(ax[0, 0], img, "Original image")
    show_img(ax[0, 1], noisy_sp, f"Salt & pepper noisy image\nPSNR = {psnr(img, noisy_sp):.2f} dB")
    positions = [(0, 2), (1, 0), (1, 1), (1, 2)]
    for (r, c), kind in zip(positions, kinds):
        out, p = results[kind]
        show_img(ax[r, c], out, f"{kind} filter\nPSNR = {p:.2f} dB")
    save(fig, out_dir, "exp2b_saltpepper_filters.png")


# --------------------------------------------------------------------------
# Optional PyQt5 desktop GUI
# The GUI only *calls* the functions above; none of the algorithms is reimplemented.
# --------------------------------------------------------------------------
def launch_gui(out_dir="Assignment3_output", cameraman_path=None, lenna_path=None):
    """Open a PyQt5 window with one tab per experiment (started by --gui)."""
    try:
        from PyQt5.QtCore import Qt, QTimer
        from PyQt5.QtGui import QImage, QPixmap
        from PyQt5.QtWidgets import (
            QApplication,
            QComboBox,
            QDoubleSpinBox,
            QFileDialog,
            QGridLayout,
            QGroupBox,
            QHBoxLayout,
            QLabel,
            QMainWindow,
            QMessageBox,
            QPushButton,
            QScrollArea,
            QSpinBox,
            QTabWidget,
            QVBoxLayout,
            QWidget,
        )
    except ImportError:
        sys.exit("Error: PyQt5 is required for --gui.  Install it with:  pip install PyQt5")

    TILE = 320  # side of every image tile in pixels
    script_dir = os.path.dirname(os.path.abspath(__file__))
    os.makedirs(out_dir, exist_ok=True)

    # ---- small helpers -----------------------------------------------------
    def to_pixmap(img, size=TILE):
        """BGR uint8 (or gray uint8) numpy image -> scaled QPixmap, aspect ratio kept."""
        if img.ndim == 2:
            arr = np.ascontiguousarray(img)
            h, w = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_Grayscale8).copy()
        else:
            arr = np.ascontiguousarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
            h, w, _ = arr.shape
            qimg = QImage(arr.data, w, h, arr.strides[0], QImage.Format_RGB888).copy()
        return QPixmap.fromImage(qimg).scaled(
            size, size, Qt.KeepAspectRatio, Qt.SmoothTransformation
        )

    def make_tile(title=""):
        """A caption + image placeholder, arranged vertically."""
        tile = QWidget()
        lay = QVBoxLayout(tile)
        lay.setContentsMargins(4, 4, 4, 4)
        lay.setSpacing(4)
        cap = QLabel(title)
        cap.setAlignment(Qt.AlignHCenter | Qt.AlignVCenter)
        cap.setWordWrap(True)
        cap.setMinimumHeight(36)
        cap.setStyleSheet("font-weight: 600;")
        pic = QLabel("not computed yet")
        pic.setFixedSize(TILE, TILE)
        pic.setAlignment(Qt.AlignCenter)
        pic.setStyleSheet("background: #202124; border: 1px solid #3c4043; color: #8a8a8a;")
        lay.addWidget(cap)
        lay.addWidget(pic)
        return {"widget": tile, "caption": cap, "picture": pic}

    def update_tile(tile, img, title):
        tile["caption"].setText(title)
        tile["picture"].setPixmap(to_pixmap(img))
        tile["picture"].setText("")

    def clear_tile(tile, text="—"):
        tile["caption"].setText("")
        tile["picture"].clear()
        tile["picture"].setText(text)

    def build_grid(tiles, columns):
        grid = QGridLayout()
        grid.setSpacing(8)
        for i, tile in enumerate(tiles):
            grid.addWidget(tile["widget"], i // columns, i % columns)
        return grid

    def scrollable(widget):
        area = QScrollArea()
        area.setWidgetResizable(True)
        area.setWidget(widget)
        return area

    def guarded(fn, parent, title):
        """Run a computation with a wait cursor; show an error dialog instead of crashing."""
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            return fn()
        except Exception as exc:  # noqa: BLE001 - keep the GUI alive on any failure
            QMessageBox.critical(parent, title, str(exc))
            return None
        finally:
            QApplication.restoreOverrideCursor()

    def ask_image(parent):
        path, _ = QFileDialog.getOpenFileName(
            parent,
            "Open image",
            script_dir,
            "Images (*.png *.jpg *.jpeg *.tif *.tiff *.bmp);;All files (*)",
        )
        if not path:
            return None
        img = cv2.imread(path, cv2.IMREAD_COLOR)
        if img is None:
            QMessageBox.critical(parent, "Open image", f"Could not read:\n{path}")
        return img

    def default_image(path, stems, demo_fn):
        """CLI path -> file next to the script -> synthetic demo image."""
        if path:
            img = cv2.imread(path, cv2.IMREAD_COLOR)
            if img is not None:
                return img
            print(f"Warning: could not read '{path}', using the default image.")
        found = find_image(script_dir, stems)
        if found:
            return load_color(found)
        return demo_fn()

    def describe(img):
        return f"{img.shape[1]} × {img.shape[0]} px"

    # ---- computations (same calls as experiment_1 / experiment_2) ----------
    def compute_exp1(img, cutoff, kind):
        F = to_frequency(img)
        spec_orig, ref_max = spectrum_to_u8(F)

        H_low = lowpass_mask(img.shape, cutoff, kind)
        H_high = highpass_mask(img.shape, cutoff, kind)
        F_low = F * H_low
        F_high = F * H_high
        spec_low, _ = spectrum_to_u8(F_low, ref_max)
        spec_high, _ = spectrum_to_u8(F_high, ref_max)

        low_raw = to_spatial(F_low)
        high_raw = to_spatial(F_high)
        low_img = to_u8(low_raw)
        high_img = scale_abs_to_u8(high_raw)

        energy = float(np.sum(np.abs(F) ** 2))
        return {
            "original": img,
            "spectrum": spec_orig,
            "mask_low": to_u8(H_low[..., 0] * 255),
            "mask_high": to_u8(H_high[..., 0] * 255),
            "spectrum_low": spec_low,
            "spectrum_high": spec_high,
            "low": low_img,
            "high": high_img,
            "energy_low": 100 * float(np.sum(np.abs(F_low) ** 2)) / energy,
            "energy_high": 100 * float(np.sum(np.abs(F_high) ** 2)) / energy,
            "mean_low": float(low_raw.mean()),
            "std_low": float(low_raw.std()),
            "psnr_low": psnr(img, low_img),
        }

    def compute_exp2a(img, sigma, s_max):
        noisy = add_gaussian_noise(img, sigma)
        restored = adaptive_median(noisy, s_max)
        return {
            "original": img,
            "noisy": noisy,
            "restored": restored,
            "psnr_noisy": psnr(img, noisy),
            "psnr_restored": psnr(img, restored),
        }

    def compute_exp2b(img, amount, k):
        noisy = add_salt_pepper(img, amount)
        kinds = ["Arithmetic mean", "Median", "Max", "Min"]
        results = {kind: window_filter(noisy, kind, k) for kind in kinds}
        return {
            "original": img,
            "noisy": noisy,
            "kinds": kinds,
            "results": results,
            "psnr_noisy": psnr(img, noisy),
            "psnr": {kind: psnr(img, results[kind]) for kind in kinds},
        }

    # ---- state + window ----------------------------------------------------
    state = {
        "cameraman": default_image(cameraman_path, ["cameraman", "Cameraman"], make_demo_cameraman),
        "lenna": default_image(lenna_path, ["lenna", "Lenna"], make_demo_scene),
        "exp1": None,
        "exp2a": None,
        "exp2b": None,
    }

    app = QApplication.instance() or QApplication([sys.argv[0]])
    app.setStyle("Fusion")

    win = QMainWindow()
    win.setWindowTitle("DIP Lab 3 — Frequency-domain filtering & noise removal")
    win.resize(1280, 900)
    tabs = QTabWidget()
    win.setCentralWidget(tabs)
    win.statusBar().showMessage(f"Output folder: {os.path.abspath(out_dir)}")

    # ======================================================================
    # Experiment 1 tab
    # ======================================================================
    tab1 = QWidget()
    tab1_layout = QVBoxLayout(tab1)

    controls1 = QHBoxLayout()
    cutoff_spin = QSpinBox()
    cutoff_spin.setRange(1, 512)
    cutoff_spin.setValue(30)
    cutoff_spin.setSuffix(" px")
    cutoff_spin.setFixedWidth(110)
    kind_combo = QComboBox()
    kind_combo.addItems(["ideal", "gaussian"])
    info1 = QLabel()
    load1_btn = QPushButton("Load image…")
    run1_btn = QPushButton("Run experiment 1")
    save1_btn = QPushButton("Save results")
    for btn in (load1_btn, run1_btn, save1_btn):
        btn.setMinimumHeight(30)

    controls1.addWidget(QLabel("Cutoff D0:"))
    controls1.addWidget(cutoff_spin)
    controls1.addSpacing(12)
    controls1.addWidget(QLabel("Filter shape:"))
    controls1.addWidget(kind_combo)
    controls1.addSpacing(12)
    controls1.addWidget(info1)
    controls1.addStretch(1)
    controls1.addWidget(load1_btn)
    controls1.addWidget(run1_btn)
    controls1.addWidget(save1_btn)
    tab1_layout.addLayout(controls1)

    metrics1 = QLabel("Press “Run experiment 1” to compute the DFT and both filters.")
    metrics1.setWordWrap(True)
    metrics1.setStyleSheet("font-family: monospace;")
    tab1_layout.addWidget(metrics1)

    tiles1 = [make_tile() for _ in range(9)]
    tab1_layout.addLayout(build_grid(tiles1, 3))
    tab1_layout.addStretch(1)
    clear_tile(tiles1[2], "")

    def fill_exp1(res):
        images = [
            res["original"],
            res["spectrum"],
            None,
            res["mask_low"],
            res["spectrum_low"],
            res["low"],
            res["mask_high"],
            res["spectrum_high"],
            res["high"],
        ]
        titles = [
            "Original image",
            "Original spectrum (log magnitude)",
            None,
            "Low-pass filter (LPF)",
            "Spectrum after LPF",
            "Spatial domain after LPF",
            "High-pass filter (HPF)",
            "Spectrum after HPF",
            "Spatial domain after HPF (|result|, scaled)",
        ]
        for tile, img, title in zip(tiles1, images, titles):
            if img is None:
                clear_tile(tile, "")
            else:
                update_tile(tile, img, title)

        metrics1.setText(
            f"Energy kept by LPF: {res['energy_low']:.2f} %    "
            f"Energy kept by HPF: {res['energy_high']:.2f} %\n"
            f"LPF result: mean = {res['mean_low']:.1f}, std = {res['std_low']:.1f}, "
            f"PSNR vs original = {res['psnr_low']:.2f} dB"
        )

    def run_exp1():
        img = state["cameraman"]
        if img is None:
            QMessageBox.warning(win, "No image", "Load an image first.")
            return
        cutoff, kind = cutoff_spin.value(), kind_combo.currentText()
        res = guarded(lambda: compute_exp1(img, cutoff, kind), win, "Experiment 1 failed")
        if res is None:
            return
        state["exp1"] = res
        fill_exp1(res)
        win.statusBar().showMessage(
            f"Experiment 1 done — {kind} filter, cutoff D0 = {cutoff} px."
        )

    def load_exp1():
        img = ask_image(win)
        if img is None:
            return
        state["cameraman"] = img
        state["exp1"] = None
        info1.setText(describe(img))
        run_exp1()

    def save_exp1():
        res = state["exp1"]
        if res is None:
            QMessageBox.information(win, "Nothing to save", "Run experiment 1 first.")
            return
        save_img(res["original"], out_dir, "exp1", "original")
        save_img(res["spectrum"], out_dir, "exp1", "spectrum_original")
        save_img(res["mask_low"], out_dir, "exp1", "lpf_mask")
        save_img(res["mask_high"], out_dir, "exp1", "hpf_mask")
        save_img(res["spectrum_low"], out_dir, "exp1", "spectrum_after_lpf")
        save_img(res["spectrum_high"], out_dir, "exp1", "spectrum_after_hpf")
        save_img(res["low"], out_dir, "exp1", "lpf_result")
        save_img(res["high"], out_dir, "exp1", "hpf_result")
        win.statusBar().showMessage(f"Saved experiment 1 images to '{os.path.join(out_dir, 'exp1')}'.")

    info1.setText(describe(state["cameraman"]))
    load1_btn.clicked.connect(load_exp1)
    run1_btn.clicked.connect(run_exp1)
    save1_btn.clicked.connect(save_exp1)

    # ======================================================================
    # Experiment 2 tab
    # ======================================================================
    tab2 = QWidget()
    tab2_layout = QVBoxLayout(tab2)

    top2 = QHBoxLayout()
    info2 = QLabel()
    load2_btn = QPushButton("Load image…")
    save2_btn = QPushButton("Save results")
    for btn in (load2_btn, save2_btn):
        btn.setMinimumHeight(30)
    top2.addWidget(info2)
    top2.addStretch(1)
    top2.addWidget(load2_btn)
    top2.addWidget(save2_btn)
    tab2_layout.addLayout(top2)

    # ---- 2a: Gaussian noise + adaptive median ----
    group_a = QGroupBox("2a — Gaussian noise → adaptive median filter")
    group_a_layout = QVBoxLayout(group_a)
    row_a = QHBoxLayout()
    sigma_spin = QSpinBox()
    sigma_spin.setRange(1, 150)
    sigma_spin.setValue(20)
    sigma_spin.setFixedWidth(90)
    smax_combo = QComboBox()
    for value in (3, 5, 7, 9, 11):
        smax_combo.addItem(str(value), value)
    smax_combo.setCurrentIndex(2)  # 7
    run_a_btn = QPushButton("Run 2a")
    run_a_btn.setMinimumHeight(30)
    row_a.addWidget(QLabel("Noise σ:"))
    row_a.addWidget(sigma_spin)
    row_a.addSpacing(12)
    row_a.addWidget(QLabel("Adaptive median S_max:"))
    row_a.addWidget(smax_combo)
    row_a.addSpacing(12)
    row_a.addWidget(run_a_btn)
    row_a.addStretch(1)
    group_a_layout.addLayout(row_a)

    metrics_a = QLabel("Press “Run 2a” to add Gaussian noise and restore the image.")
    metrics_a.setStyleSheet("font-family: monospace;")
    metrics_a.setWordWrap(True)
    group_a_layout.addWidget(metrics_a)

    tiles_a = [make_tile() for _ in range(3)]
    group_a_layout.addLayout(build_grid(tiles_a, 3))
    tab2_layout.addWidget(group_a)

    # ---- 2b: Salt & pepper noise + mean / median / max / min ----
    group_b = QGroupBox("2b — Salt & pepper noise → arithmetic mean / median / max / min")
    group_b_layout = QVBoxLayout(group_b)
    row_b = QHBoxLayout()
    amount_spin = QDoubleSpinBox()
    amount_spin.setRange(0.1, 100.0)
    amount_spin.setDecimals(1)
    amount_spin.setSingleStep(1.0)
    amount_spin.setValue(5.0)
    amount_spin.setSuffix(" %")
    amount_spin.setFixedWidth(100)
    k_combo = QComboBox()
    for value in (3, 5, 7, 9):
        k_combo.addItem(f"{value}×{value}", value)
    run_b_btn = QPushButton("Run 2b")
    run_b_btn.setMinimumHeight(30)
    row_b.addWidget(QLabel("Noise amount:"))
    row_b.addWidget(amount_spin)
    row_b.addSpacing(12)
    row_b.addWidget(QLabel("Filter window:"))
    row_b.addWidget(k_combo)
    row_b.addSpacing(12)
    row_b.addWidget(run_b_btn)
    row_b.addStretch(1)
    group_b_layout.addLayout(row_b)

    metrics_b = QLabel("Press “Run 2b” to add salt & pepper noise and compare the four filters.")
    metrics_b.setStyleSheet("font-family: monospace;")
    metrics_b.setWordWrap(True)
    group_b_layout.addWidget(metrics_b)

    tiles_b = [make_tile() for _ in range(6)]
    group_b_layout.addLayout(build_grid(tiles_b, 3))
    tab2_layout.addWidget(group_b)
    tab2_layout.addStretch(1)

    def fill_exp2a(res):
        update_tile(tiles_a[0], res["original"], "Original image")
        update_tile(
            tiles_a[1],
            res["noisy"],
            f"Noisy image (before filter)\nPSNR = {res['psnr_noisy']:.2f} dB",
        )
        update_tile(
            tiles_a[2],
            res["restored"],
            f"Adaptive median filter (after)\nPSNR = {res['psnr_restored']:.2f} dB",
        )
        metrics_a.setText(
            f"Gaussian noise: PSNR = {res['psnr_noisy']:.2f} dB    "
            f"after adaptive median: PSNR = {res['psnr_restored']:.2f} dB"
        )

    def fill_exp2b(res):
        update_tile(tiles_b[0], res["original"], "Original image")
        update_tile(
            tiles_b[1],
            res["noisy"],
            f"Salt & pepper noisy image\nPSNR = {res['psnr_noisy']:.2f} dB",
        )
        for tile, kind in zip(tiles_b[2:], res["kinds"]):
            update_tile(
                tile,
                res["results"][kind],
                f"{kind} filter\nPSNR = {res['psnr'][kind]:.2f} dB",
            )
        metrics_b.setText(
            "    ".join(f"{kind}: {res['psnr'][kind]:.2f} dB" for kind in res["kinds"])
        )

    def run_exp2a():
        img = state["lenna"]
        if img is None:
            QMessageBox.warning(win, "No image", "Load an image first.")
            return
        sigma, s_max = sigma_spin.value(), smax_combo.currentData()
        res = guarded(lambda: compute_exp2a(img, sigma, s_max), win, "Experiment 2a failed")
        if res is None:
            return
        state["exp2a"] = res
        fill_exp2a(res)
        win.statusBar().showMessage(
            f"Experiment 2a done — σ = {sigma}, S_max = {s_max}."
        )

    def run_exp2b():
        img = state["lenna"]
        if img is None:
            QMessageBox.warning(win, "No image", "Load an image first.")
            return
        amount = amount_spin.value() / 100.0
        k = k_combo.currentData()
        res = guarded(lambda: compute_exp2b(img, amount, k), win, "Experiment 2b failed")
        if res is None:
            return
        state["exp2b"] = res
        fill_exp2b(res)
        win.statusBar().showMessage(
            f"Experiment 2b done — {100 * amount:g} % salt & pepper, {k}×{k} windows."
        )

    def load_exp2():
        img = ask_image(win)
        if img is None:
            return
        state["lenna"] = img
        state["exp2a"] = None
        state["exp2b"] = None
        info2.setText(describe(img))
        run_exp2a()
        run_exp2b()

    def save_exp2():
        a, b = state["exp2a"], state["exp2b"]
        if a is None and b is None:
            QMessageBox.information(win, "Nothing to save", "Run experiment 2 first.")
            return
        if a is not None:
            save_img(a["original"], out_dir, "exp2", "original")
            save_img(a["noisy"], out_dir, "exp2", "gaussian_noisy")
            save_img(a["restored"], out_dir, "exp2", "gaussian_adaptive_median")
        if b is not None:
            save_img(b["noisy"], out_dir, "exp2", "saltpepper_noisy")
            for kind in b["kinds"]:
                save_img(
                    b["results"][kind],
                    out_dir,
                    "exp2",
                    "saltpepper_" + kind.lower().replace(" ", "_"),
                )
        win.statusBar().showMessage(f"Saved experiment 2 images to '{os.path.join(out_dir, 'exp2')}'.")

    info2.setText(describe(state["lenna"]))
    load2_btn.clicked.connect(load_exp2)
    run_a_btn.clicked.connect(run_exp2a)
    run_b_btn.clicked.connect(run_exp2b)
    save2_btn.clicked.connect(save_exp2)

    # ---- menu bar ----------------------------------------------------------
    file_menu = win.menuBar().addMenu("&File")
    file_menu.addAction("Open cameraman image for experiment 1…", load_exp1)
    file_menu.addAction("Open Lenna image for experiment 2…", load_exp2)
    file_menu.addSeparator()
    file_menu.addAction("Save experiment 1 results", save_exp1)
    file_menu.addAction("Save experiment 2 results", save_exp2)
    file_menu.addSeparator()
    file_menu.addAction("Quit", win.close)

    tabs.addTab(scrollable(tab1), "Experiment 1 — Frequency filtering")
    tabs.addTab(scrollable(tab2), "Experiment 2 — Noise removal")

    win.show()
    # Populate the tabs once the event loop starts so the window appears immediately.
    # Staggered timers keep the order deterministic and let the window repaint between runs.
    QTimer.singleShot(0, run_exp1)
    QTimer.singleShot(50, run_exp2a)
    QTimer.singleShot(100, run_exp2b)
    return app.exec_()


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Assignment 3: frequency-domain filtering & noise removal (RGB)")
    p.add_argument(
        "--cameraman", help="path to the cameraman image (default: cameraman.* next to the script, else a synthetic image)"
    )
    p.add_argument(
        "--lenna", help="path to the Lenna image (default: lenna.* next to the script, else a synthetic image)"
    )
    p.add_argument(
        "--cutoff",
        type=float,
        default=30.0,
        help="cutoff radius D0 in frequency samples for Exp 1 (default 30)",
    )
    p.add_argument(
        "--filter",
        choices=["ideal", "gaussian"],
        default="ideal",
        help="shape of the low-pass / high-pass filter for Exp 1 (default ideal)",
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="std of the Gaussian noise for Exp 2a (default 20)",
    )
    p.add_argument(
        "--smax",
        type=int,
        default=7,
        help="largest window size of the adaptive median filter, odd (default 7)",
    )
    p.add_argument(
        "--sp",
        type=float,
        default=0.05,
        help="fraction of values corrupted by salt & pepper noise in Exp 2b (default 0.05)",
    )
    p.add_argument(
        "--ksize",
        type=int,
        default=3,
        help="window size of the mean / median / max / min filters, odd (default 3)",
    )
    p.add_argument("--out", default="Assignment3_output", help="output folder for figures")
    p.add_argument(
        "--no-show", action="store_true", help="save figures without opening windows"
    )
    p.add_argument(
        "--gui",
        action="store_true",
        help="open the PyQt5 desktop GUI (uses --cameraman, --lenna and --out when given)",
    )
    args = p.parse_args()

    if args.gui:
        os.makedirs(args.out, exist_ok=True)
        launch_gui(args.out, args.cameraman, args.lenna)
        return

    if args.cutoff <= 0:
        sys.exit("Error: --cutoff must be > 0")
    if args.sigma <= 0:
        sys.exit("Error: --sigma must be > 0")
    if not (0 < args.sp <= 1):
        sys.exit("Error: --sp must be in (0, 1]")
    if args.ksize < 3 or args.ksize % 2 == 0:
        sys.exit("Error: --ksize must be an odd number >= 3")
    if args.smax < 3 or args.smax % 2 == 0:
        sys.exit("Error: --smax must be an odd number >= 3")

    os.makedirs(args.out, exist_ok=True)

    print("Loading images...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    cameraman = get_image(
        args.cameraman, ["cameraman", "Cameraman"], script_dir, make_demo_cameraman, args.out
    )
    lenna = get_image(
        args.lenna, ["lenna", "Lenna"], script_dir, make_demo_scene, args.out
    )

    experiment_1(cameraman, args.cutoff, args.filter, args.out)
    experiment_2(lenna, args.sigma, args.sp, args.ksize, args.smax, args.out)
    print(f"\nDone. Figures and result images are in '{args.out}'.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()