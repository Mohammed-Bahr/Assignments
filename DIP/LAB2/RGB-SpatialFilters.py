#!/usr/bin/env python3
"""
Digital Image Processing - Assignment 2
Spatial Filters (RGB / color version)

Experiment 1: Gaussian noise, then smoothing with Average / Median / Gaussian filters (3x3 and 5x5)
Experiment 2: Edge detection (Prewitt, Sobel, Laplacian), then sharpening with each edge result

Color handling: every filter is applied independently to each of the B, G, R channels
(this is what OpenCV does natively for multi-channel images). Images are kept in OpenCV's
BGR order internally and converted to RGB only for display with matplotlib.

Usage:
    python SpitalFilters.py                                   # looks for veggies.png / sunflower.png, else synthetic images
    python SpitalFilters.py --veggies veggies.png --sunflower sunflower.png
    python SpitalFilters.py --sigma 25 --amount 1.5
    python SpitalFilters.py --no-show                         # only save figures (no pop-up windows)

Figures and result images are saved to ./Assignment2_output (change with --out).
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


def get_image(path, default_path, demo_fn, out_dir):
    """Use the given path, else default_path if it exists, else a synthetic demo image."""
    if path:
        return load_color(path)
    if os.path.exists(default_path):
        print(f"  using '{default_path}'")
        return load_color(default_path)

    name = os.path.basename(default_path)
    print(f"  '{name}' not found -> using a synthetic demo image.")
    img = demo_fn()
    cv2.imwrite(os.path.join(out_dir, "demo_" + name), img)
    return img


def colorize(gray, gains_bgr):
    """Turn a gray demo image into a BGR image by scaling each channel (only used for demo images)."""
    img = gray[..., None].astype(np.float64) * np.array(gains_bgr, dtype=np.float64)
    return np.clip(img, 0, 255).astype(np.uint8)


def make_demo_scene(h=300, w=400, seed=0):
    """Synthetic image with smooth areas, sharp edges and some texture (stand-in for veggies.png)."""
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


def make_demo_flower(h=300, w=400):
    """Synthetic 'flower' (stand-in for sunflower.png): petals + centre + stem on a plain background."""
    y, x = np.mgrid[0:h, 0:w].astype(np.float64)
    cx, cy = w / 2, h * 0.42
    dx, dy = x - cx, y - cy
    r = np.hypot(dx, dy)
    theta = np.arctan2(dy, dx)
    petals = r < (70 + 30 * np.cos(12 * theta))
    centre = r < 40
    stem = (abs(x - cx) < 6) & (y > cy)
    img = np.full((h, w), 70.0)
    img[stem] = 110
    img[petals] = 200
    img[centre] = 50 + 15 * np.sin(r[centre] / 2.5)  # textured centre
    img = cv2.GaussianBlur(img, (0, 0), 1.0)  # slightly soft edges, like a real photo
    gray = np.clip(img, 0, 255).astype(np.uint8)
    return colorize(gray, (0.4, 0.9, 1.0))  # yellowish tint (B, G, R gains)


def add_gaussian_noise(img, sigma, mean=0.0, seed=0):
    """Add Gaussian noise N(mean, sigma^2) and clip back to [0, 255].
    img.shape includes the channel axis, so every channel gets its own independent noise."""
    rng = np.random.default_rng(seed)
    noise = rng.normal(mean, sigma, img.shape)
    return np.clip(img.astype(np.float64) + noise, 0, 255).astype(np.uint8) # saturation which forces the pixels to stay between 0 and 255


def mse(a, b):
    """Mean squared error between two images (averaged over all pixels and channels)."""
    return float(np.mean((a.astype(np.float64) - b.astype(np.float64)) ** 2))


def psnr(a, b):
    """Peak signal-to-noise ratio in dB (higher = closer to the reference)."""
    m = mse(a, b)
    return float("inf") if m == 0 else 10 * np.log10(255.0**2 / m)



# --------------------------------------------------------------------------
# Filters  (all of them act channel-by-channel on (M, N, 3) images)
# --------------------------------------------------------------------------
def smooth(img, kind, k):
    """Smoothing filter of window size k x k: 'Average', 'Median' or 'Gaussian'."""
    if kind == "Average":
        return cv2.blur(img, (k, k))
    if kind == "Median":
        return cv2.medianBlur(img, k)
    if kind == "Gaussian":
        return cv2.GaussianBlur(img, (k, k), 0)  # sigma computed from the window size
    raise ValueError(kind)


# Prewitt kernels (OpenCV has no built-in Prewitt, so we use filter2D)
PREWITT_X = np.array([[-1, 0, 1], [-1, 0, 1], [-1, 0, 1]], dtype=np.float64)
PREWITT_Y = np.array([[-1, -1, -1], [0, 0, 0], [1, 1, 1]], dtype=np.float64)


def prewitt_edges(img):
    """Gradient magnitude with Prewitt kernels, divided by 3 so it is in gray levels per pixel."""
    f = img.astype(np.float64)
    gx = cv2.filter2D(f, cv2.CV_64F, PREWITT_X)  # filter2D filters each channel separately
    gy = cv2.filter2D(f, cv2.CV_64F, PREWITT_Y)
    return np.hypot(gx, gy)


# Sobel 3x3 kernels (center-weighted gradient approximations)
SOBEL_X = np.array([[-1, 0, 1], [-2, 0, 2], [-1, 0, 1]], dtype=np.float64)
SOBEL_Y = np.array([[-1, -2, -1], [0, 0, 0], [1, 2, 1]], dtype=np.float64)


def sobel_edges(img):
    """Gradient magnitude with Sobel 3x3 kernels (normalized by 1/4 to gray levels per pixel)."""
    f = img.astype(np.float64)
    gx = cv2.filter2D(f, cv2.CV_64F, SOBEL_X)
    gy = cv2.filter2D(f, cv2.CV_64F, SOBEL_Y)
    return np.hypot(gx, gy)


def laplacian_signed(img):
    """Signed Laplacian (4-neighbour kernel, centre = -4). Its sign matters for sharpening."""
    return cv2.Laplacian(img.astype(np.float64), cv2.CV_64F, ksize=1)


def edge_to_u8(edges):
    """Scale an edge magnitude for display without letting a few strong edges dominate.
    One common scale is used for all channels so edge colors are preserved."""
    magnitude = np.abs(edges).astype(np.float64)
    limit = float(np.percentile(magnitude, 99))
    if limit <= 0:
        return np.zeros(magnitude.shape, dtype=np.uint8)
    return np.clip(magnitude * (255.0 / limit), 0, 255).astype(np.uint8)


def sharpen_with_gradient(img, grad, amount):
    """Sharpening with a gradient-magnitude edge map: s = r + amount * edges (clipped)."""
    return np.clip(img.astype(np.float64) + amount * grad, 0, 255).astype(np.uint8)


def sharpen_with_laplacian(img, lap, amount):
    """Sharpening with the Laplacian (centre = -4): s = r - amount * Laplacian (clipped)."""
    return np.clip(img.astype(np.float64) - amount * lap, 0, 255).astype(np.uint8)


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
def experiment_1(img, sigma, out_dir):
    print("\n=== Experiment 1: Gaussian noise & smoothing filters ===")
    noisy = add_gaussian_noise(img, sigma)
    print(f"  image size: {img.shape[1]}x{img.shape[0]}   noise: Gaussian, mean=0, sigma={sigma}")
    print(f"  noisy image vs original: PSNR = {psnr(img, noisy):.2f} dB")

    save_img(img, out_dir, "exp1", "original")
    save_img(noisy, out_dir, "exp1", "noisy")

    kinds = ["Average", "Median", "Gaussian"]
    sizes = [3, 5]
    results = {}  # (kind, k) -> (image, psnr)
    for k in sizes:
        for kind in kinds:
            out = smooth(noisy, kind, k)
            results[(kind, k)] = (out, psnr(img, out))
            print(f"  {kind:<8} {k}x{k}: PSNR = {results[(kind, k)][1]:.2f} dB")
            save_img(out, out_dir, "exp1", f"{kind.lower()}_{k}x{k}")

    # single grid: row 0 = original / noisy, row 1 = 3x3 filters, row 2 = 5x5 filters
    fig, ax = plt.subplots(3, 3, figsize=(13, 9), constrained_layout=True)
    fig.suptitle(f"Experiment 1: Gaussian noise (σ = {sigma:g}) and smoothing", fontsize=15)
    show_img(ax[0, 0], img, "Original image")
    show_img(ax[0, 1], noisy, f"Noisy image\nPSNR = {psnr(img, noisy):.2f} dB")
    ax[0, 2].axis("off")
    for row, k in enumerate(sizes, start=1):
        for col, kind in enumerate(kinds):
            out, p = results[(kind, k)]
            show_img(ax[row, col], out, f"{kind} {k}x{k}\nPSNR = {p:.2f} dB")
    save(fig, out_dir, "exp1_smoothing.png")


def experiment_2(img, amount, out_dir):
    print("\n=== Experiment 2: edge detection & sharpening ===")
    prew = prewitt_edges(img)
    sob = sobel_edges(img)
    lap = laplacian_signed(img)

    sharp_p = sharpen_with_gradient(img, prew, amount)
    sharp_s = sharpen_with_gradient(img, sob, amount)
    sharp_l = sharpen_with_laplacian(img, lap, amount)

    for name, e in (("Prewitt", prew), ("Sobel", sob), ("Laplacian", np.abs(lap))):
        print(f"  {name:<9} edge map: mean={e.mean():.2f}  max={e.max():.1f}")
    print(f"  sharpening amount = {amount}")
    for name, s in (("Prewitt", sharp_p), ("Sobel", sharp_s), ("Laplacian", sharp_l)):
        print(f"  sharpened with {name:<9}: mean={s.mean():.1f}  std={s.std():.1f}  (original std={img.std():.1f})")

    save_img(img, out_dir, "exp2", "original")
    prew_display = edge_to_u8(prew)
    sob_display = edge_to_u8(sob)
    lap_display = edge_to_u8(lap)
    save_img(prew_display, out_dir, "exp2", "edges_prewitt")
    save_img(sob_display, out_dir, "exp2", "edges_sobel")
    save_img(lap_display, out_dir, "exp2", "edges_laplacian")
    save_img(sharp_p, out_dir, "exp2", "sharpened_prewitt")
    save_img(sharp_s, out_dir, "exp2", "sharpened_sobel")
    save_img(sharp_l, out_dir, "exp2", "sharpened_laplacian")

    # single grid: top = original + edge maps, bottom = sharpened versions
    fig, ax = plt.subplots(2, 4, figsize=(18, 9), constrained_layout=True)
    fig.suptitle(f"Experiment 2: Edge detection and sharpening (amount = {amount:g})", fontsize=15)
    show_img(ax[0, 0], img, "Original image")
    show_img(ax[0, 1], prew_display, "Prewitt edge magnitude (scaled)")
    show_img(ax[0, 2], sob_display, "Sobel edge magnitude (scaled)")
    show_img(ax[0, 3], lap_display, "Laplacian magnitude (scaled)")
    ax[1, 0].axis("off")
    show_img(ax[1, 1], sharp_p, "Sharpened (Prewitt)")
    show_img(ax[1, 2], sharp_s, "Sharpened (Sobel)")
    show_img(ax[1, 3], sharp_l, "Sharpened (Laplacian)")
    save(fig, out_dir, "exp2_edges_sharpening.png")


# --------------------------------------------------------------------------
def main():
    p = argparse.ArgumentParser(description="Assignment 2: spatial filters (RGB)")
    p.add_argument(
        "--veggies", help="path to veggies.png (default: ./veggies.png, else a synthetic image)"
    )
    p.add_argument(
        "--sunflower", help="path to sunflower.png (default: ./sunflower.png, else a synthetic image)"
    )
    p.add_argument(
        "--sigma",
        type=float,
        default=20.0,
        help="std of the Gaussian noise for Exp 1 (default 20)",
    )
    p.add_argument(
        "--amount",
        type=float,
        default=1.0,
        help="sharpening strength for Exp 2 (default 1.0)",
    )
    p.add_argument("--out", default="Assignment2_output", help="output folder for figures")
    p.add_argument(
        "--no-show", action="store_true", help="save figures without opening windows"
    )
    args = p.parse_args()

    if args.sigma <= 0:
        sys.exit("Error: --sigma must be > 0")
    if args.amount < 0:
        sys.exit("Error: --amount must be >= 0")

    os.makedirs(args.out, exist_ok=True)

    print("Loading images...")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    veggies = get_image(
        args.veggies, os.path.join(script_dir, "veggies.png"), make_demo_scene, args.out
    )
    sunflower = get_image(
        args.sunflower,
        os.path.join(script_dir, "sunflower.png"),
        make_demo_flower,
        args.out,
    )

    experiment_1(veggies, args.sigma, args.out)
    experiment_2(sunflower, args.amount, args.out)
    print(f"\nDone. Figures and result images are in '{args.out}'.")

    if not args.no_show:
        plt.show()


if __name__ == "__main__":
    main()
