from pathlib import Path
import json
import numpy as np
import tifffile as tiff

import matplotlib
matplotlib.use("TkAgg")  # stable interactive backend without Qt
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector

from skimage.filters import threshold_otsu
from skimage.restoration import denoise_nl_means
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing


# =========================
# SETTINGS (PORTABLE)
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
TIFF_FOLDER = PROJECT_ROOT / "example_dataset" / "Raw_TIFF_Slices"

# Optional filter; keep "" to load all tif files in folder
FILENAME_KEYWORD = ""

# IMPORTANT: set to None to auto-load ALL slices in the folder
MAX_SLICES = None

# Start slice
SLICE_INDEX = 0

# ---- Crop (ACTIVE) ----
# Crop is always enabled. Drag a rectangle on the LEFT image to set crop.
CROP_R0, CROP_R1 = 0, None
CROP_C0, CROP_C1 = 0, None

# ---- Circle (set in code) ----
CENTER_X = 1480
CENTER_Y = 1480
RADIUS = 1250
RADIUS_SCROLL_STEP = 20  # scroll wheel changes radius

OUTSIDE_VALUE = 0

# ---- Denoise (fixed; not interactive) ----
DENOISE_MODE = "gaussian"     # "none" | "gaussian" | "median" | "nlm"
DENOISE_STRENGTH = 1.5

# ---- Threshold (ACTIVE) ----
THRESH_MODE = "manual"        # "manual" | "otsu"
MANUAL_THRESH = 140.0
THRESH_STEP = 5.0             # [ and ] adjust manual threshold

# Pores definition (fixed)
PORES_DARK = True

# Cleanup (fixed; not interactive)
OPEN_R = 1
CLOSE_R = 0

# ---- Save (ONLY when you press G) ----
# Root folder; script will create: outputs/<dataset_name>/
SAVE_ROOT = PROJECT_ROOT / "outputs"
SAVE_TAG = "run"
LATEST_POINTER_JSON = SAVE_ROOT / "_LATEST_PARAMS.json"
# =========================


def dataset_name_from_folder(folder: Path) -> str:
    name = Path(folder).name
    if not name:
        name = "dataset"
    keep = []
    for ch in name:
        if ch.isalnum() or ch in ("-", "_"):
            keep.append(ch)
        else:
            keep.append("_")
    return "".join(keep)


def list_tifs(folder, keyword=None):
    folder = Path(folder)
    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if keyword:
        kw = str(keyword).lower()
        files = [f for f in files if kw in f.name.lower()]
    return files


def load_stack(folder, keyword=None, max_slices=None):
    files = list_tifs(folder, keyword=keyword)
    if not files:
        raise RuntimeError(f"No tif files found in {folder} with keyword={keyword}")

    if max_slices is not None:
        files = files[:int(max_slices)]

    vol = np.stack([tiff.imread(str(f)) for f in files], axis=0)
    return vol, files


def normalize01(img):
    img = img.astype(np.float32)
    mn, mx = np.nanmin(img), np.nanmax(img)
    if mx <= mn:
        return np.zeros_like(img, dtype=np.float32)
    return (img - mn) / (mx - mn)


def circle_mask(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def apply_mask(img, mask, outside_value=0):
    out = img.copy()
    out[~mask] = outside_value
    return out


def denoise(img, mode, strength):
    imgf = img.astype(np.float32)
    if mode == "none":
        return imgf
    if mode == "gaussian":
        return gaussian_filter(imgf, sigma=max(0.0, float(strength)))
    if mode == "median":
        k = int(max(1, strength))
        if k % 2 == 0:
            k += 1
        return median_filter(imgf, size=k)
    h = max(0.0, float(strength))
    return denoise_nl_means(imgf, h=h, patch_size=5, patch_distance=6, fast_mode=True)


def segment_pores(img, mask, pores_dark, thresh_mode, manual_t):
    core = img[mask]
    if core.size == 0:
        return np.zeros_like(mask, dtype=bool), float("nan")
    t = threshold_otsu(core) if thresh_mode == "otsu" else float(manual_t)
    pore = (img < t) if pores_dark else (img > t)
    pore &= mask
    return pore, float(t)


def cleanup(mask, open_r, close_r):
    out = mask
    if open_r > 0:
        out = binary_opening(out, structure=disk(int(open_r)))
    if close_r > 0:
        out = binary_closing(out, structure=disk(int(close_r)))
    return out


def overlay_cyan(gray01, mask, alpha=0.55):
    rgb = np.dstack([gray01, gray01, gray01])
    rgb[..., 0] = np.where(mask, (1 - alpha) * rgb[..., 0] + alpha * 0.0, rgb[..., 0])
    rgb[..., 1] = np.where(mask, (1 - alpha) * rgb[..., 1] + alpha * 1.0, rgb[..., 1])
    rgb[..., 2] = np.where(mask, (1 - alpha) * rgb[..., 2] + alpha * 1.0, rgb[..., 2])
    return rgb


def ensure_dir(p):
    Path(p).mkdir(parents=True, exist_ok=True)


def save_outputs(save_folder, tag, params, crop01, masked01, den01, mask_u8, overlay_rgb):
    save_folder = Path(save_folder)
    ensure_dir(save_folder)

    params_path = save_folder / f"{tag}_params.json"
    with open(params_path, "w", encoding="utf-8") as f:
        json.dump(params, f, indent=2)

    plt.imsave(save_folder / f"{tag}_01_crop.png", crop01, cmap="gray")
    plt.imsave(save_folder / f"{tag}_02_masked.png", masked01, cmap="gray")
    plt.imsave(save_folder / f"{tag}_03_denoised.png", den01, cmap="gray")
    plt.imsave(save_folder / f"{tag}_04_mask.png", mask_u8, cmap="gray", vmin=0, vmax=255)
    plt.imsave(save_folder / f"{tag}_05_overlay.png", overlay_rgb)

    ensure_dir(LATEST_POINTER_JSON.parent)
    with open(LATEST_POINTER_JSON, "w", encoding="utf-8") as f:
        json.dump({"latest_params_json": str(params_path)}, f, indent=2)

    return params_path


def main():
    print(f"Running from: {PROJECT_ROOT}")
    vol, files = load_stack(TIFF_FOLDER, keyword=FILENAME_KEYWORD, max_slices=MAX_SLICES)
    z_count, height, width = vol.shape

    ds_name = dataset_name_from_folder(TIFF_FOLDER)
    save_folder = SAVE_ROOT / ds_name

    state = {
        "slice": int(np.clip(SLICE_INDEX, 0, z_count - 1)),
        "crop_r0": int(CROP_R0),
        "crop_r1": height if CROP_R1 is None else int(CROP_R1),
        "crop_c0": int(CROP_C0),
        "crop_c1": width if CROP_C1 is None else int(CROP_C1),
        "cx": int(CENTER_X if CENTER_X is not None else width // 2),
        "cy": int(CENTER_Y if CENTER_Y is not None else height // 2),
        "r": int(RADIUS if RADIUS is not None else (min(height, width) // 2 - 10)),
        "denoise_mode": str(DENOISE_MODE),
        "denoise_strength": float(DENOISE_STRENGTH),
        "thresh_mode": str(THRESH_MODE),
        "thresh": float(MANUAL_THRESH),
        "pores_dark": bool(PORES_DARK),
        "open_r": int(OPEN_R),
        "close_r": int(CLOSE_R),
        "outside_value": int(OUTSIDE_VALUE),
    }

    fig = plt.figure(figsize=(16, 7))
    ax_left = fig.add_axes([0.05, 0.10, 0.44, 0.82])
    ax_right = fig.add_axes([0.52, 0.10, 0.44, 0.82])
    for ax in (ax_left, ax_right):
        ax.axis("off")

    im_left = ax_left.imshow(np.zeros((10, 10), dtype=np.float32), cmap="gray")
    im_right = ax_right.imshow(np.zeros((10, 10, 3), dtype=np.float32))
    circle_line_left, = ax_left.plot([], [], lw=2)

    info = fig.text(
        0.05, 0.95,
        f"Dataset: {ds_name} | Drag crop on LEFT | scroll radius | t toggle otsu/manual | [ ] threshold | ←/→ slice | g save",
        fontsize=11
    )

    def draw_circle_on(line, cx, cy, r):
        theta = np.linspace(0, 2 * np.pi, 360)
        xs = cx + r * np.cos(theta)
        ys = cy + r * np.sin(theta)
        line.set_data(xs, ys)

    def compute_current():
        z = int(np.clip(state["slice"], 0, z_count - 1))
        raw_full = vol[z]

        r0, r1 = state["crop_r0"], state["crop_r1"]
        c0, c1 = state["crop_c0"], state["crop_c1"]

        r0 = max(0, min(height - 1, r0))
        c0 = max(0, min(width - 1, c0))
        r1 = max(r0 + 1, min(height, r1))
        c1 = max(c0 + 1, min(width, c1))

        crop = raw_full[r0:r1, c0:c1]
        crop01 = normalize01(crop)

        cx_crop = state["cx"] - c0
        cy_crop = state["cy"] - r0
        mask = circle_mask(crop.shape[0], crop.shape[1], cy_crop, cx_crop, state["r"])

        masked = apply_mask(crop, mask, outside_value=state["outside_value"])
        masked01 = normalize01(masked)

        den = denoise(masked, state["denoise_mode"], state["denoise_strength"])
        den = apply_mask(den, mask, outside_value=state["outside_value"])
        den01 = normalize01(den)

        pm, used_t = segment_pores(
            den, mask,
            pores_dark=state["pores_dark"],
            thresh_mode=state["thresh_mode"],
            manual_t=state["thresh"]
        )
        pm = cleanup(pm, state["open_r"], state["close_r"])
        pm &= mask

        overlay = overlay_cyan(masked01, pm)
        mask_u8 = pm.astype(np.uint8) * 255
        porosity = pm.sum() / max(mask.sum(), 1)

        meta = {
            "slice": z,
            "crop_box": [int(r0), int(r1), int(c0), int(c1)],
            "circle_full": [int(state["cx"]), int(state["cy"]), int(state["r"])],
            "circle_crop": [int(cx_crop), int(cy_crop), int(state["r"])],
            "used_t": float(used_t),
            "porosity": float(porosity),
        }
        return crop01, masked01, den01, overlay, mask_u8, meta

    def redraw():
        crop01, masked01, den01, overlay, mask_u8, meta = compute_current()

        im_left.set_data(crop01)
        ax_left.set_title(f"LEFT: Crop view (slice {meta['slice']}) | drag crop | scroll radius")

        cx_crop, cy_crop, r = meta["circle_crop"]
        draw_circle_on(circle_line_left, cx_crop, cy_crop, r)

        im_right.set_data(overlay)
        ax_right.set_title(
            f"RIGHT: Overlay | mode={state['thresh_mode']} | used_t={meta['used_t']:.1f} | porosity={meta['porosity']:.4f}"
        )

        info.set_text(
            f"Dataset: {ds_name} | slices loaded={z_count} | ←/→ slice | drag crop | wheel radius | t toggle otsu/manual | [ ] threshold | g save\n"
            f"Slice {meta['slice']} | crop_box={meta['crop_box']} | circle(cx={state['cx']}, cy={state['cy']}, r={state['r']}) | "
            f"thresh_mode={state['thresh_mode']} manual={state['thresh']:.1f} used_t={meta['used_t']:.1f}"
        )

        fig.canvas.draw_idle()

    def on_crop_select(eclick, erelease):
        x0, y0 = eclick.xdata, eclick.ydata
        x1, y1 = erelease.xdata, erelease.ydata
        if x0 is None or x1 is None or y0 is None or y1 is None:
            return

        c0, c1 = sorted([int(round(x0)), int(round(x1))])
        r0, r1 = sorted([int(round(y0)), int(round(y1))])

        cur_r0, cur_r1 = state["crop_r0"], state["crop_r1"]
        cur_c0, cur_c1 = state["crop_c0"], state["crop_c1"]

        cur_h = cur_r1 - cur_r0
        cur_w = cur_c1 - cur_c0

        c0 = max(0, min(cur_w - 1, c0))
        c1 = max(1, min(cur_w, c1))
        r0 = max(0, min(cur_h - 1, r0))
        r1 = max(1, min(cur_h, r1))

        state["crop_r0"] = cur_r0 + r0
        state["crop_r1"] = cur_r0 + r1
        state["crop_c0"] = cur_c0 + c0
        state["crop_c1"] = cur_c0 + c1

        redraw()

    rect_sel = RectangleSelector(
        ax_left, on_crop_select,
        useblit=True, button=[1], interactive=True
    )
    rect_sel.set_active(True)

    def on_scroll(event):
        if event.inaxes != ax_left:
            return
        step = int(RADIUS_SCROLL_STEP)
        if event.button == "up":
            state["r"] = int(state["r"] + step)
        elif event.button == "down":
            state["r"] = int(max(10, state["r"] - step))
        redraw()

    def on_key(event):
        if event.key is None:
            return

        if event.key == "right":
            state["slice"] = min(state["slice"] + 1, z_count - 1)
            redraw()
            return
        if event.key == "left":
            state["slice"] = max(state["slice"] - 1, 0)
            redraw()
            return

        if event.key.lower() == "t":
            state["thresh_mode"] = "otsu" if state["thresh_mode"] == "manual" else "manual"
            redraw()
            return

        if event.key == "[":
            state["thresh"] = max(0.0, float(state["thresh"]) - float(THRESH_STEP))
            redraw()
            return
        if event.key == "]":
            state["thresh"] = min(65535.0, float(state["thresh"]) + float(THRESH_STEP))
            redraw()
            return

        if event.key.lower() == "g":
            crop01, masked01, den01, overlay, mask_u8, meta = compute_current()

            params = {
                "dataset_name": ds_name,
                "tiff_folder": str(TIFF_FOLDER),
                "filename_keyword": FILENAME_KEYWORD,
                "num_slices_found": int(z_count),
                "max_slices_loaded": int(z_count),
                "slice": int(meta["slice"]),

                "crop_r0": int(state["crop_r0"]),
                "crop_r1": int(state["crop_r1"]),
                "crop_c0": int(state["crop_c0"]),
                "crop_c1": int(state["crop_c1"]),

                "cx": int(state["cx"]),
                "cy": int(state["cy"]),
                "r": int(state["r"]),
                "outside_value": int(state["outside_value"]),

                "denoise_mode": state["denoise_mode"],
                "denoise_strength": float(state["denoise_strength"]),

                "thresh_mode": state["thresh_mode"],
                "manual_thresh": float(state["thresh"]),
                "used_threshold": float(meta["used_t"]),
                "pores_dark": bool(state["pores_dark"]),

                "open_r": int(state["open_r"]),
                "close_r": int(state["close_r"]),

                "porosity_preview_slice": float(meta["porosity"]),
            }

            tag = f"{ds_name}_{SAVE_TAG}_z{meta['slice']}"
            params_path = save_outputs(save_folder, tag, params, crop01, masked01, den01, mask_u8, overlay)
            info.set_text(
                f"Saved params+images to: {save_folder}\n"
                f"Latest params pointer: {LATEST_POINTER_JSON}\n"
                f"Params: {params_path}"
            )
            fig.canvas.draw_idle()
            return

    fig.canvas.mpl_connect("scroll_event", on_scroll)
    fig.canvas.mpl_connect("key_press_event", on_key)

    redraw()
    plt.show()


if __name__ == "__main__":
    main()