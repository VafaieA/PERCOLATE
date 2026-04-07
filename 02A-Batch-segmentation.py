from pathlib import Path
import json
import numpy as np
import tifffile as tiff

from skimage.filters import threshold_otsu
from skimage.morphology import disk
from scipy.ndimage import gaussian_filter, median_filter, binary_opening, binary_closing
from skimage.restoration import denoise_nl_means


# =========================
# SETTINGS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
TIFF_FOLDER = PROJECT_ROOT / "example_dataset" / "Raw_TIFF_Slices"

# Where Script 1 saved params
PARAMS_JSON = PROJECT_ROOT / "outputs" / "_LATEST_PARAMS.json"

# Output
OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_NAME = "mask_stack.tif"
# =========================


def load_latest_params(pointer_json):
    pointer_json = Path(pointer_json)
    if not pointer_json.exists():
        raise FileNotFoundError(f"Pointer file not found: {pointer_json}")

    with open(pointer_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    params_path = data.get("latest_params_json", None)
    if not params_path or not Path(params_path).exists():
        raise FileNotFoundError(f"Params JSON not found: {params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params


def list_tifs(folder):
    folder = Path(folder)
    files = sorted(list(folder.glob("*.tif")) + list(folder.glob("*.tiff")))
    if not files:
        raise RuntimeError(f"No tif files found in {folder}")
    return files


def circle_mask(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def denoise(img, mode, strength):
    imgf = img.astype(np.float32)

    if mode == "none":
        return imgf
    if mode == "gaussian":
        return gaussian_filter(imgf, sigma=float(strength))
    if mode == "median":
        k = int(max(1, strength))
        if k % 2 == 0:
            k += 1
        return median_filter(imgf, size=k)

    return denoise_nl_means(imgf, h=float(strength), fast_mode=True)


def cleanup(mask, open_r, close_r):
    if open_r > 0:
        mask = binary_opening(mask, structure=disk(open_r))
    if close_r > 0:
        mask = binary_closing(mask, structure=disk(close_r))
    return mask


def main():
    print(f"Running from: {PROJECT_ROOT}")
    params = load_latest_params(PARAMS_JSON)

    files = list_tifs(TIFF_FOLDER)
    print(f"Found {len(files)} slices")

    masks = []

    for i, fpath in enumerate(files):
        img = tiff.imread(str(fpath))

        # Crop
        r0, r1 = params["crop_r0"], params["crop_r1"]
        c0, c1 = params["crop_c0"], params["crop_c1"]
        img = img[r0:r1, c0:c1]

        # Circle
        cx = params["cx"] - c0
        cy = params["cy"] - r0
        r = params["r"]

        mask_circle = circle_mask(img.shape[0], img.shape[1], cy, cx, r)

        img_masked = img.copy()
        img_masked[~mask_circle] = params["outside_value"]

        # Denoise
        img_denoised = denoise(
            img_masked,
            params["denoise_mode"],
            params["denoise_strength"]
        )
        img_denoised[~mask_circle] = params["outside_value"]

        # Threshold
        if params["thresh_mode"] == "otsu":
            core = img_denoised[mask_circle]
            thresh = threshold_otsu(core)
        else:
            thresh = params["manual_thresh"]

        if params["pores_dark"]:
            pore_mask = img_denoised < thresh
        else:
            pore_mask = img_denoised > thresh

        pore_mask &= mask_circle

        # Cleanup
        pore_mask = cleanup(
            pore_mask,
            params["open_r"],
            params["close_r"]
        )

        masks.append(pore_mask.astype(np.uint8) * 255)

        if i % 50 == 0:
            print(f"Processed {i}/{len(files)}")

    masks = np.stack(masks, axis=0)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_path = OUTPUT_DIR / OUTPUT_NAME

    tiff.imwrite(str(out_path), masks)

    print(f"\nSaved mask stack to: {out_path}")


if __name__ == "__main__":
    main()