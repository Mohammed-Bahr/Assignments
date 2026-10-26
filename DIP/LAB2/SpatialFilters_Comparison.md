# `SpatialFilters.py` vs `RGB-SpatialFilters.py`

Both scripts are the same assignment (LAB 2): Experiment 1 adds Gaussian noise and then
smooths with Average / Median / Gaussian kernels at 3×3 and 5×5; Experiment 2 computes
Prewitt, Sobel and Laplacian edges and sharpens the image with each edge map. The CLI
flags (`--veggies`, `--sunflower`, `--sigma`, `--amount`, `--out`, `--no-show`), the
experiment functions, the plotting grids and the output file names are identical.

The difference is **one axis of the data model: the number of channels.**

| | `SpatialFilters.py` | `RGB-SpatialFilters.py` |
|---|---|---|
| Image shape | `(M, N)` | `(M, N, 3)` |
| Loader | `load_gray` → `IMREAD_GRAYSCALE` | `load_color` → `IMREAD_COLOR` (BGR order) |
| `get_image` param | `default_name` | `default_path` (only the demo filename is derived, via `basename`) |
| Demo images | Grayscale numpy scenes | Same scenes, then passed through `colorize()` to get a tint |
| Channel axis | none | B, G, R, handled independently everywhere |

Everything else follows from that.

## 1. Loading

`SpatialFilters.py` decodes straight to 8-bit grayscale, so a real `veggies.png` loses its
colour before any processing. `RGB-SpatialFilters.py` uses `cv2.imread(path, cv2.IMREAD_COLOR)`,
which keeps three channels in OpenCV's **BGR** order (not RGB). The docstring is explicit that
BGR is kept internally and converted to RGB only for display.

## 2. Synthetic demo images

The demo generators are the *same code*, with one addition at the end of each:

```python
# grayscale version
return np.clip(img, 0, 255).astype(np.uint8)

# RGB version
gray = np.clip(img, 0, 255).astype(np.uint8)
return colorize(gray, (0.7, 1.0, 0.8))  # greenish tint (B, G, R gains)
```

`colorize(gray, gains_bgr)` is a new helper that stacks the single-channel image three times
and multiplies each channel by a constant gain, clipping to `[0, 255]`. `make_demo_scene`
uses `(0.7, 1.0, 0.8)` (greenish) and `make_demo_flower` uses `(0.4, 0.9, 1.0)` (yellowish).
This is a cheap way to get *some* colour into the fallback images, but note the underlying
structure is still identical in all three channels — it is a tint, not real per-channel
content. Real photos are the only way to see genuine colour-channel behaviour.

## 3. Noise is added per channel, independently

`add_gaussian_noise` is textually the same function, but `rng.normal(mean, sigma, img.shape)`
producing `(M, N, 3)` draws **three independent noise fields**: the blue noise, green noise and
red noise are uncorrelated. The consequence is coloured (chromatic) speckle in the RGB version,
whereas the grayscale version gets neutral grey speckle.

## 4. PSNR / MSE are no longer directly comparable

`mse` averages over *all* pixels **and all channels** (`np.mean` over the whole array). That makes
the RGB PSNR a channel-weighted average on the same 0–255 scale, so it is still meaningful within
the RGB script, but its numeric values will not line up with the grayscale script's values for the
same `--sigma`, even on the same source image.

## 5. Filters run channel-by-channel

`smooth()` is byte-for-byte identical: `cv2.blur`, `cv2.medianBlur` and `cv2.GaussianBlur` all
process a 3-channel `uint8` array by applying the kernel independently to each channel. Same for
`cv2.filter2D` in `prewitt_edges` / `sobel_edges` and for `cv2.Laplacian`. The RGB file just adds
explanatory comments (`# filter2D filters each channel separately`).

This is a real methodological choice, not a formality: filtering B, G and R separately is **not**
the same as converting to luminance, filtering, and keeping the chroma. Independent per-channel
filtering can shift hue at edges (colour fringing) and a channel that is nearly flat contributes
noise-dominated edges to the combined result.

## 6. Edge maps carry colour

In the grayscale version `np.hypot(gx, gy)` returns a single `(M, N)` magnitude map. In the RGB
version the same expression returns `(M, N, 3)` — a per-channel gradient magnitude. `np.abs(lap)`
is likewise three-channel. So the "edge image" in the RGB script is a colour image showing which
channel had strong gradients where, instead of a single luminance edge strength.

## 7. Display scaling of edges: one shared scale

`edge_to_u8` is the same code in both files. Because `np.percentile` runs over the whole array in
the RGB case, the 99th-percentile limit is **shared by all three channels** (the docstring states
this: "One common scale is used for all channels so edge colors are preserved"). A per-channel
scale would blow each channel's edges up to white independently and destroy the relative colour
information. So the same line behaves differently, on purpose.

## 8. Colour-order conversions

- **Plotting:** `show_img` is the only function with a real branch. For `ndim == 3` it does
  `cv2.cvtColor(img, cv2.COLOR_BGR2RGB)` before `ax.imshow`; the grayscale file always uses
  `cmap="gray", vmin=0, vmax=255`. Without this conversion the plots would show blue vegetables.
- **Saving:** `save_img` is unchanged and needs no conversion, because `cv2.imwrite` already
  expects BGR.
- **Sharpening:** `sharpen_with_gradient` and `sharpen_with_laplacian` are identical
  (`clip(img ± amount * edges)`). In the RGB case they add the per-channel edge values to each
  channel independently — gradient sharpening is applied to an unsigned magnitude, so both
  scripts brighten edges rather than performing a true deconvolution-style unsharp mask.

## 9. Practical caveat: same output folder

Neither script varies `--out`, which defaults to `Assignment2_output`, and the per-image names
(`exp1/original.png`, `exp2/sharpened_sobel.png`, …) and figure names are the same. Running both
scripts in the same directory silently **overwrites** the earlier run's results. Pass different
`--out` values when you want to keep both, e.g.

```bash
python SpatialFilters.py     --out out_gray --no-show
python RGB-SpatialFilters.py --out out_rgb  --no-show
```

## Summary

`RGB-SpatialFilters.py` is the grayscale script with the channel axis threaded through the whole
pipeline: colour loading, per-channel noise, channel-wise filtering, three-channel edge maps, a
shared edge display scale, and BGR→RGB conversion for matplotlib. The kernels, the experiments,
the metrics and the CLI are unchanged. The interesting side effect is that the RGB version is
*more* than three independent grayscale runs, because the per-channel noise is uncorrelated and
the per-channel edge magnitudes get recombined into a single colour edge image.
