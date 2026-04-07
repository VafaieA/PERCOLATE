from pathlib import Path
import json
import csv
import numpy as np
import tifffile as tiff
from scipy.ndimage import label, find_objects

# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_POINTER_JSON = PROJECT_ROOT / "outputs" / "_LATEST_PARAMS.json"

# Input mask stack from Script 02A
MASK_STACK_CANDIDATES = [
    PROJECT_ROOT / "outputs" / "mask_stack.tif",
]

# Output root
OUT_ROOT = PROJECT_ROOT / "outputs"

# =========================
# DATASET SELECTION
# =========================
# None = use latest dataset name from params
DATASET_NAME = None

# =========================
# VOXEL SIZE (µm)
# =========================
VOXEL_UM_XY = 7.14224
VOXEL_UM_Z = 7.14224

# =========================
# PREVIEW DOWNSAMPLING FOR VISUALIZER
# =========================
# This creates pore_preview_3d.tif used by Script 03
PREVIEW_DS_Z = 6
PREVIEW_DS_Y = 2
PREVIEW_DS_X = 2

# =========================
# CONNECTIVITY / FILTERING
# =========================
USE_26_CONNECTIVITY = True
MIN_COMPONENT_VOXELS = 1

# =========================
# SIZE METHOD
# =========================
# "xy_area_est"   -> 2D-like equivalent radius
# "sphere_volume" -> 3D sphere-equivalent radius
SIZE_METHOD = "xy_area_est"

# =========================
# HISTOGRAM BINS (RADIUS, µm)
# =========================
HIST_BINS_UM = [0, 5, 10, 20, 40, 80, 120, 200, 500]


def load_latest_params():
    if not LATEST_POINTER_JSON.exists():
        raise FileNotFoundError(
            f"Missing pointer file:\n{LATEST_POINTER_JSON}\n"
            f"Run Script 1 and press G first."
        )

    with open(LATEST_POINTER_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)

    params_path = d.get("latest_params_json", "")
    if not params_path or not Path(params_path).exists():
        raise FileNotFoundError(f"Invalid params JSON path:\n{params_path}")

    with open(params_path, "r", encoding="utf-8") as f:
        params = json.load(f)

    return params_path, params


def resolve_dataset_name(params):
    if DATASET_NAME is not None and str(DATASET_NAME).strip():
        return str(DATASET_NAME).strip()

    ds = params.get("dataset_name", "")
    if ds:
        return ds

    tiff_folder = params.get("tiff_folder", "")
    return Path(tiff_folder).name or "dataset"


def find_mask_stack():
    for p in MASK_STACK_CANDIDATES:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Could not find mask_stack.tif.\nTried:\n" + "\n".join(str(p) for p in MASK_STACK_CANDIDATES)
    )


def ensure_dir(path):
    Path(path).mkdir(parents=True, exist_ok=True)


def circle_mask(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def r_eq_um_from_component(nvox, z_extent, voxel_area_xy_um2, voxel_vol_um3):
    if SIZE_METHOD.lower() == "sphere_volume":
        vol_um3 = float(nvox) * float(voxel_vol_um3)
        return (3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    z_extent = max(int(z_extent), 1)
    area_xy_vox = float(nvox) / float(z_extent)
    area_xy_um2 = area_xy_vox * float(voxel_area_xy_um2)
    return np.sqrt(area_xy_um2 / np.pi)


def save_hist_csv(csv_path, hist_edges, hist_counts):
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["bin_lo_um", "bin_hi_um", "count"])
        for i in range(len(hist_counts)):
            writer.writerow([hist_edges[i], hist_edges[i + 1], int(hist_counts[i])])


def load_mask_stack_safely(mask_stack_path):
    """
    Try memory-mapped access first to avoid loading everything into RAM.
    Falls back to normal read if needed.
    """
    try:
        arr = tiff.memmap(str(mask_stack_path))
        return arr, True
    except Exception:
        arr = tiff.imread(str(mask_stack_path))
        return arr, False


def main():
    print(f"Running from: {PROJECT_ROOT}")
    params_json, params = load_latest_params()
    dataset_name = resolve_dataset_name(params)

    mask_stack_path = find_mask_stack()
    out_dir = OUT_ROOT / dataset_name
    ensure_dir(out_dir)

    pore_preview_tif = out_dir / "pore_preview_3d.tif"
    hist_csv_path = out_dir / "pore_eq_radius_hist_um.csv"
    summary_json_path = out_dir / "summary.json"
    components_csv_path = out_dir / "pore_components.csv"

    print("\n=== PATHS ===")
    print("Project root :", PROJECT_ROOT)
    print("Dataset      :", dataset_name)
    print("Params JSON  :", params_json)
    print("Mask stack   :", mask_stack_path)
    print("Out dir      :", out_dir)
    print("==================\n")

    stack_u8, is_memmap = load_mask_stack_safely(mask_stack_path)

    if stack_u8.ndim != 3:
        raise RuntimeError(f"Expected 3D mask stack, got shape {stack_u8.shape}")

    z_full, y_full, x_full = stack_u8.shape

    print(f"[INFO] Stack shape: {(z_full, y_full, x_full)}")
    print(f"[INFO] Access mode: {'memmap' if is_memmap else 'in-memory'}")

    crop_r0 = int(params.get("crop_r0", 0))
    crop_r1 = int(params.get("crop_r1", y_full))
    crop_c0 = int(params.get("crop_c0", 0))
    crop_c1 = int(params.get("crop_c1", x_full))

    cx_full = int(params["cx"])
    cy_full = int(params["cy"])
    r_full = int(params["r"])

    cx_crop = cx_full - crop_c0
    cy_crop = cy_full - crop_r0

    core2d = circle_mask(y_full, x_full, cy_crop, cx_crop, r_full)
    core_voxels = int(core2d.sum()) * int(z_full)

    if core_voxels <= 0:
        raise RuntimeError("Core mask has zero voxels.")

    z_idx = np.arange(0, z_full, PREVIEW_DS_Z)
    y_idx = np.arange(0, y_full, PREVIEW_DS_Y)
    x_idx = np.arange(0, x_full, PREVIEW_DS_X)

    z_prev = len(z_idx)
    y_prev = len(y_idx)
    x_prev = len(x_idx)

    print(f"[INFO] Preview shape will be: {(z_prev, y_prev, x_prev)}")
    print(f"[INFO] Preview downsample (Z,Y,X): ({PREVIEW_DS_Z}, {PREVIEW_DS_Y}, {PREVIEW_DS_X})")

    core2d_preview = core2d[::PREVIEW_DS_Y, ::PREVIEW_DS_X]

    pore_preview = np.zeros((z_prev, y_prev, x_prev), dtype=np.uint8)
    pore_voxels_full = 0

    for out_k, src_k in enumerate(z_idx):
        sl = stack_u8[src_k] > 0
        pore_voxels_full += int(np.count_nonzero(sl & core2d))

        sl_preview = sl[::PREVIEW_DS_Y, ::PREVIEW_DS_X]
        sl_preview_in_core = sl_preview & core2d_preview
        pore_preview[out_k] = sl_preview_in_core.astype(np.uint8) * 255

        if out_k % 50 == 0 or out_k == z_prev - 1:
            print(f"[INFO] Processed preview slice {out_k + 1}/{z_prev}")

    porosity_total = float(pore_voxels_full) / float(core_voxels)

    print(f"[INFO] Core voxels (full): {core_voxels}")
    print(f"[INFO] Pore voxels in core (full): {pore_voxels_full}")
    print(f"[INFO] Total porosity (full, slice-wise): {100.0 * porosity_total:.3f}%")

    tiff.imwrite(str(pore_preview_tif), pore_preview)
    print(f"[INFO] Saved preview pore mask: {pore_preview_tif}")

    pores_preview_bool = pore_preview > 0

    if USE_26_CONNECTIVITY:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        connectivity_name = "26-neighbor"
    else:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, 1] = 1
        structure[0, 1, 1] = structure[2, 1, 1] = 1
        structure[1, 0, 1] = structure[1, 2, 1] = 1
        structure[1, 1, 0] = structure[1, 1, 2] = 1
        connectivity_name = "6-neighbor"

    lbl, n = label(pores_preview_bool.astype(np.uint8), structure=structure)
    counts = np.bincount(lbl.ravel())
    if counts.size == 0:
        counts = np.array([0], dtype=np.int64)

    objs = find_objects(lbl)
    valid = np.where(counts >= MIN_COMPONENT_VOXELS)[0]
    valid = valid[valid != 0]

    print(f"[INFO] Connected components found on preview: {n}")
    print(f"[INFO] Components kept on preview: {len(valid)}")
    print(f"[INFO] Connectivity: {connectivity_name}")

    voxel_xy_preview_um = float(VOXEL_UM_XY) * float(PREVIEW_DS_X)
    voxel_z_preview_um = float(VOXEL_UM_Z) * float(PREVIEW_DS_Z)

    voxel_area_xy_um2 = voxel_xy_preview_um * voxel_xy_preview_um
    voxel_vol_um3 = voxel_z_preview_um * voxel_xy_preview_um * voxel_xy_preview_um

    radii_um = []
    component_rows = []

    for lab_id in valid:
        slc = objs[lab_id - 1] if (lab_id - 1) < len(objs) else None
        z_extent = 1 if slc is None else int(slc[0].stop - slc[0].start)

        nvox = int(counts[lab_id])
        r_um = r_eq_um_from_component(nvox, z_extent, voxel_area_xy_um2, voxel_vol_um3)

        radii_um.append(r_um)
        component_rows.append({
            "label": int(lab_id),
            "nvox_preview": nvox,
            "z_extent_preview": int(z_extent),
            "r_eq_um": float(r_um),
        })

    radii_um = np.asarray(radii_um, dtype=float)

    if radii_um.size > 0:
        print(f"[INFO] r_eq min/max on preview ({SIZE_METHOD}): {radii_um.min():.3f} / {radii_um.max():.3f} µm")
    else:
        print("[INFO] No preview components available for radius statistics.")

    hist_edges = np.asarray(HIST_BINS_UM, dtype=float)
    if hist_edges.ndim != 1 or hist_edges.size < 2:
        raise ValueError("HIST_BINS_UM must contain at least two increasing edges.")
    if not np.all(np.diff(hist_edges) > 0):
        raise ValueError("HIST_BINS_UM must be strictly increasing.")

    hist_counts, hist_edges = np.histogram(radii_um, bins=hist_edges)
    save_hist_csv(hist_csv_path, hist_edges, hist_counts)
    print(f"[INFO] Saved histogram CSV: {hist_csv_path}")

    with open(components_csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["label", "nvox_preview", "z_extent_preview", "r_eq_um"])
        for row in component_rows:
            writer.writerow([
                row["label"],
                row["nvox_preview"],
                row["z_extent_preview"],
                row["r_eq_um"],
            ])
    print(f"[INFO] Saved component table: {components_csv_path}")

    summary = {
        "dataset_name": dataset_name,
        "project_root": str(PROJECT_ROOT),
        "params_json": str(params_json),
        "mask_stack_path": str(mask_stack_path),
        "out_dir": str(out_dir),

        "full_volume_shape_zyx": [int(z_full), int(y_full), int(x_full)],
        "preview_volume_shape_zyx": [int(z_prev), int(y_prev), int(x_prev)],

        "crop_box": [crop_r0, crop_r1, crop_c0, crop_c1],
        "circle_full": {
            "cx": int(cx_full),
            "cy": int(cy_full),
            "r": int(r_full),
        },
        "circle_crop": {
            "cx": int(cx_crop),
            "cy": int(cy_crop),
            "r": int(r_full),
        },

        "voxel_um_original": {
            "xy": float(VOXEL_UM_XY),
            "z": float(VOXEL_UM_Z),
        },
        "preview_downsample_zyx": [int(PREVIEW_DS_Z), int(PREVIEW_DS_Y), int(PREVIEW_DS_X)],
        "voxel_um_preview_effective": {
            "xy": float(voxel_xy_preview_um),
            "z": float(voxel_z_preview_um),
        },

        "connectivity": connectivity_name,
        "size_method": SIZE_METHOD,
        "min_component_voxels": int(MIN_COMPONENT_VOXELS),

        "core_voxels_full": int(core_voxels),
        "pore_voxels_in_core_full": int(pore_voxels_full),
        "porosity_total_full": float(porosity_total),

        "clustering_performed_on": "preview_volume_only",
        "n_components_total_preview": int(n),
        "n_components_kept_preview": int(len(valid)),

        "files": {
            "pore_preview_3d_tif": str(pore_preview_tif),
            "hist_csv": str(hist_csv_path),
            "components_csv": str(components_csv_path),
        }
    }

    if radii_um.size > 0:
        summary["r_eq_um_stats_preview"] = {
            "min": float(np.min(radii_um)),
            "max": float(np.max(radii_um)),
            "mean": float(np.mean(radii_um)),
            "median": float(np.median(radii_um)),
        }
    else:
        summary["r_eq_um_stats_preview"] = {
            "min": None,
            "max": None,
            "mean": None,
            "median": None,
        }

    with open(summary_json_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    print(f"[INFO] Saved summary JSON: {summary_json_path}")
    print("\nDone.")


if __name__ == "__main__":
    main()