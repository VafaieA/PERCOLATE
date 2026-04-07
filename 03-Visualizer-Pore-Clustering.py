from pathlib import Path
import json
import numpy as np
import tifffile as tiff
from scipy.ndimage import label, find_objects
from skimage.measure import marching_cubes

try:
    import napari
except ImportError as e:
    raise ImportError(
        "Napari could not start because Qt bindings are missing.\n\n"
        "Install in your virtual environment with:\n"
        "    python -m pip install pyqt5\n"
        "or:\n"
        '    python -m pip install "napari[pyqt5]"\n'
    ) from e


# =========================
# PATHS
# =========================
PROJECT_ROOT = Path(__file__).resolve().parent
LATEST_POINTER_JSON = PROJECT_ROOT / "outputs" / "_LATEST_PARAMS.json"

OUT_ROOT_CANDIDATES = [
    PROJECT_ROOT / "outputs",
]

# =========================
# DATASET SELECTION
# =========================
DATASET_NAME = None

# =========================
# CLASS BINS (RADIUS, µm)
# =========================
USER_BINS_UM = [0, 20, 40, 80, 120, 500]

# =========================
# FALLBACK VOXEL SIZE (µm)
# =========================
VOXEL_UM_XY = 7.14224
VOXEL_UM_Z = 7.14224

# =========================
# INPUT VOLUME CHOICE
# =========================
PREFER_FULL_RES_MASK = False

PREVIEW_BASE_DS_Z = 6
PREVIEW_BASE_DS_Y = 2
PREVIEW_BASE_DS_X = 2

# Optional extra downsampling inside THIS script
# Keep these at 1 for final figures
DS_Z = 1
DS_Y = 1
DS_X = 1

# =========================
# COMPONENTS / FILTERS
# =========================
USE_26_CONNECTIVITY_FOR_COMPONENTS = True
MIN_COMPONENT_VOXELS = 1

# "xy_area_est"   -> approximate 2D-like equivalent radius
# "sphere_volume" -> 3D sphere-equivalent radius
SIZE_METHOD = "xy_area_est"

# =========================
# DISPLAY
# =========================
SHOW_CORE_SURFACE = True
CORE_RGB = (0.80, 0.80, 0.80)
CORE_OPACITY = 0.05

SURFACE_OPACITY = 0.90
MC_LEVEL = 0.5
MAX_FACES_PER_LAYER = 900_000

BINARY_IMAGE_OPACITY = 0.50
BINARY_IMAGE_RENDERING = "mip"
BINARY_IMAGE_BLENDING = "translucent_no_depth"
BINARY_IMAGE_CONTRAST_LIMITS = (0, 1)

NAPARI_THEME = "light"
SHOW_AXES = False
SHOW_GRID = False
SHOW_SCALE_BAR = True


def fmt_plain(x, nd=2):
    try:
        x = float(x)
    except Exception:
        return str(x)
    if not np.isfinite(x):
        return str(x)
    s = f"{x:.{nd}f}".rstrip("0").rstrip(".")
    return s if s else "0"


def circle_mask(h, w, cy, cx, r):
    y, x = np.ogrid[:h, :w]
    return (y - cy) ** 2 + (x - cx) ** 2 <= r ** 2


def class_colors_for_pores():
    return [
        (0.00, 0.30, 1.00),  # blue
        (0.00, 0.75, 0.20),  # green
        (1.00, 0.90, 0.00),  # yellow
        (1.00, 0.50, 0.00),  # orange
        (1.00, 0.00, 0.00),  # red
    ]


def solid_colormap(rgb):
    r, g, b = rgb
    return {
        "colors": np.array([[r, g, b, 1.0], [r, g, b, 1.0]], dtype=np.float32),
        "name": f"solid_{int(r*255)}_{int(g*255)}_{int(b*255)}",
        "interpolation": "linear",
    }


def add_surface_solid(viewer, verts, faces, rgb, name, opacity):
    values = np.ones((verts.shape[0],), dtype=np.float32)
    viewer.add_surface(
        (verts, faces, values),
        name=name,
        opacity=float(opacity),
        colormap=solid_colormap(rgb),
    )


def add_binary_image_solid(viewer, vol_u8, rgb, name, scale_um):
    layer = viewer.add_image(
        vol_u8,
        name=name,
        scale=scale_um,
        opacity=float(BINARY_IMAGE_OPACITY),
        rendering=BINARY_IMAGE_RENDERING,
        blending=BINARY_IMAGE_BLENDING,
        contrast_limits=BINARY_IMAGE_CONTRAST_LIMITS,
        colormap=solid_colormap(rgb),
    )
    try:
        layer.interpolation = "nearest"
    except Exception:
        pass
    return layer


def eq_radius_um_from_component(nvox: int, z_extent: int, voxel_area_xy_um2: float, voxel_vol_um3: float) -> float:
    if SIZE_METHOD.lower() == "sphere_volume":
        vol_um3 = float(nvox) * float(voxel_vol_um3)
        return (3.0 * vol_um3 / (4.0 * np.pi)) ** (1.0 / 3.0)

    z_extent = max(int(z_extent), 1)
    area_xy_vox = float(nvox) / float(z_extent)
    area_xy_um2 = area_xy_vox * float(voxel_area_xy_um2)
    return np.sqrt(area_xy_um2 / np.pi)


def load_latest_params_json_path():
    if not LATEST_POINTER_JSON.exists():
        raise FileNotFoundError(
            f"Missing pointer file:\n{LATEST_POINTER_JSON}\n"
            f"Run Script 1 and press G first."
        )

    with open(LATEST_POINTER_JSON, "r", encoding="utf-8") as f:
        d = json.load(f)

    params_path = d.get("latest_params_json", "")
    if not params_path or not Path(params_path).exists():
        raise FileNotFoundError(
            f"Latest params JSON is missing or invalid:\n{params_path}"
        )

    return Path(params_path)


def load_latest_params():
    params_json = load_latest_params_json_path()
    with open(params_json, "r", encoding="utf-8") as f:
        params = json.load(f)
    return params_json, params


def find_dataset_dir(dataset_name):
    for root in OUT_ROOT_CANDIDATES:
        candidate = root / dataset_name
        if candidate.is_dir():
            return candidate

    for root in OUT_ROOT_CANDIDATES:
        if root.is_dir():
            return root

    return None


def resolve_dataset_paths():
    params_json = ""
    params = {}

    if DATASET_NAME is None:
        params_json, params = load_latest_params()
        dataset_name = params.get("dataset_name")
        if not dataset_name:
            dataset_name = Path(params.get("tiff_folder", "")).name or "dataset"
    else:
        dataset_name = str(DATASET_NAME).strip()

    out_dir = find_dataset_dir(dataset_name)
    if out_dir is None:
        raise FileNotFoundError(
            "Could not find any output directory.\n"
            f"Tried roots:\n" + "\n".join(str(p) for p in OUT_ROOT_CANDIDATES)
        )

    maybe_dataset_dir = out_dir / dataset_name
    if maybe_dataset_dir.is_dir():
        out_dir = maybe_dataset_dir

    pore_preview_tif = out_dir / "pore_preview_3d.tif"
    pore_full_tif = out_dir / "pore_mask_3d.tif"
    summary_json = out_dir / "summary.json"

    return params, params_json, dataset_name, out_dir, pore_preview_tif, pore_full_tif, summary_json


def choose_input_volume(pore_full_tif: Path, pore_preview_tif: Path):
    if PREFER_FULL_RES_MASK and pore_full_tif.exists():
        return pore_full_tif, (1, 1, 1), "FULL-RES (pore_mask_3d.tif)"

    if pore_preview_tif.exists():
        return pore_preview_tif, (PREVIEW_BASE_DS_Z, PREVIEW_BASE_DS_Y, PREVIEW_BASE_DS_X), "PREVIEW (pore_preview_3d.tif)"

    if pore_full_tif.exists():
        return pore_full_tif, (1, 1, 1), "FULL-RES fallback (pore_mask_3d.tif)"

    raise FileNotFoundError(
        f"Neither input file exists:\n{pore_full_tif}\n{pore_preview_tif}"
    )


def load_geometry(params, summary_json):
    if params and all(k in params for k in ["cx", "cy", "r"]):
        crop_r0 = int(params.get("crop_r0", 0))
        crop_c0 = int(params.get("crop_c0", 0))
        cx_full = int(params["cx"])
        cy_full = int(params["cy"])
        r_full = int(params["r"])
        return crop_r0, crop_c0, cx_full, cy_full, r_full

    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            ssum = json.load(f)

        crop_box = ssum.get("crop_box", [0, 0, 0, 0])
        crop_r0 = int(crop_box[0])
        crop_c0 = int(crop_box[2])

        circ = ssum.get("circle_full", {})
        cx_full = int(circ.get("cx", 0))
        cy_full = int(circ.get("cy", 0))
        r_full = int(circ.get("r", 0))
        return crop_r0, crop_c0, cx_full, cy_full, r_full

    raise FileNotFoundError(
        "Could not determine crop/circle geometry.\n"
        f"Missing summary.json:\n{summary_json}\n"
        "and no valid params were available from the latest pointer."
    )


def load_summary(summary_json):
    if summary_json.exists():
        with open(summary_json, "r", encoding="utf-8") as f:
            return json.load(f)
    return {}


def get_voxel_and_preview_settings(summary):
    voxel_xy = float(summary.get("voxel_um_original", {}).get("xy", VOXEL_UM_XY))
    voxel_z = float(summary.get("voxel_um_original", {}).get("z", VOXEL_UM_Z))

    preview_ds = summary.get("preview_downsample_zyx", [PREVIEW_BASE_DS_Z, PREVIEW_BASE_DS_Y, PREVIEW_BASE_DS_X])
    if len(preview_ds) != 3:
        preview_ds = [PREVIEW_BASE_DS_Z, PREVIEW_BASE_DS_Y, PREVIEW_BASE_DS_X]

    preview_ds_z, preview_ds_y, preview_ds_x = map(int, preview_ds)
    return voxel_xy, voxel_z, preview_ds_z, preview_ds_y, preview_ds_x


def main():
    print(f"Running from: {PROJECT_ROOT}")
    print("VISUALIZER VERSION: lightweight, vivid colors, precise scaling")

    try:
        from qtpy import QtWidgets  # noqa: F401
    except Exception as e:
        raise ImportError(
            "Qt is not installed in this environment.\n"
            "Install it with:\n"
            "python -m pip install pyqt5"
        ) from e

    params, params_json, dataset_name, out_dir, pore_preview_tif, pore_full_tif, summary_json = resolve_dataset_paths()
    summary = load_summary(summary_json)

    crop_r0, crop_c0, cx_full, cy_full, r_full = load_geometry(params, summary_json)
    cx_crop = cx_full - crop_c0
    cy_crop = cy_full - crop_r0

    voxel_xy_um, voxel_z_um, preview_ds_z, preview_ds_y, preview_ds_x = get_voxel_and_preview_settings(summary)

    input_path, base_ds_zyx, input_desc = choose_input_volume(pore_full_tif, pore_preview_tif)
    base_ds_z, base_ds_y, base_ds_x = map(int, base_ds_zyx)

    if "PREVIEW" in input_desc.upper():
        base_ds_z, base_ds_y, base_ds_x = preview_ds_z, preview_ds_y, preview_ds_x

    print("\n=== PATHS ===")
    print("Project root :", PROJECT_ROOT)
    print("Dataset      :", dataset_name)
    print("Out dir      :", out_dir)
    print("Input        :", input_path)
    print("Summary JSON :", summary_json)
    print("Params JSON  :", params_json)
    print("==================\n")

    u8 = tiff.imread(str(input_path))
    pores0 = (u8 > 0)

    pores = pores0[::DS_Z, ::DS_Y, ::DS_X]
    zp, yp, xp = pores.shape

    total_ds_z = base_ds_z * DS_Z
    total_ds_y = base_ds_y * DS_Y
    total_ds_x = base_ds_x * DS_X

    dz_um = voxel_z_um * total_ds_z
    dy_um = voxel_xy_um * total_ds_y
    dx_um = voxel_xy_um * total_ds_x
    scale_um = (dz_um, dy_um, dx_um)

    voxel_area_xy_um2 = float(dy_um) * float(dx_um)
    voxel_vol_um3 = float(dz_um) * float(dy_um) * float(dx_um)

    cx_view = int(round(cx_crop / total_ds_x))
    cy_view = int(round(cy_crop / total_ds_y))
    r_view_px = int(round(r_full / total_ds_x))

    cx_view = max(0, min(xp - 1, cx_view))
    cy_view = max(0, min(yp - 1, cy_view))
    r_view_px = max(1, r_view_px)

    core2d = circle_mask(yp, xp, cy_view, cx_view, r_view_px)
    pore_in_core = pores & core2d[None, :, :]

    core_vox_view = int(core2d.sum()) * int(zp)
    if core_vox_view <= 0:
        raise RuntimeError("Core mask has zero voxels after downsampling. Try smaller DS_* or check mapping.")

    phi_preview_total = float(pore_in_core.sum()) / float(core_vox_view)

    length_mm = (zp * dz_um) / 1000.0
    diameter_mm = (2.0 * r_full * voxel_xy_um) / 1000.0

    print(f"[INFO] Using input: {input_desc}")
    print(f"[INFO] Original voxel size (um): Z={voxel_z_um}, XY={voxel_xy_um}")
    print(f"[INFO] Downsample vs original (Z,Y,X): {(total_ds_z, total_ds_y, total_ds_x)}")
    print(f"[INFO] Effective display voxel size (um): Z={dz_um}, Y={dy_um}, X={dx_um}")
    print(f"[INFO] Approx sample length (mm): {length_mm:.3f}")
    print(f"[INFO] Approx sample diameter (mm): {diameter_mm:.3f}")
    print(f"[INFO] Average porosity (%): {100.0 * phi_preview_total:.3f}")

    if USE_26_CONNECTIVITY_FOR_COMPONENTS:
        structure = np.ones((3, 3, 3), dtype=np.uint8)
        conn_txt = "26-neighbor"
    else:
        structure = np.zeros((3, 3, 3), dtype=np.uint8)
        structure[1, 1, 1] = 1
        structure[0, 1, 1] = structure[2, 1, 1] = 1
        structure[1, 0, 1] = structure[1, 2, 1] = 1
        structure[1, 1, 0] = structure[1, 1, 2] = 1
        conn_txt = "6-neighbor"

    lbl, n = label(pore_in_core.astype(np.uint8), structure=structure)
    counts = np.bincount(lbl.ravel())
    if counts.size == 0:
        counts = np.array([0], dtype=np.int64)

    objs = find_objects(lbl)
    valid = np.where(counts >= MIN_COMPONENT_VOXELS)[0]
    valid = valid[valid != 0]
    is_valid = np.zeros(n + 1, dtype=bool)
    is_valid[valid] = True

    bins_edges = np.array(USER_BINS_UM, dtype=float)
    if bins_edges.size != 6:
        raise ValueError("USER_BINS_UM must have exactly 6 edges (5 classes).")

    for i in range(1, bins_edges.size):
        if bins_edges[i] <= bins_edges[i - 1]:
            raise ValueError("USER_BINS_UM must be strictly increasing.")

    n_classes = 5
    class_colors = class_colors_for_pores()
    label_to_class = np.full(n + 1, -1, dtype=np.int16)

    r_list = []
    for lab_id in valid:
        slc = objs[lab_id - 1] if (lab_id - 1) < len(objs) else None
        z_extent = 1 if slc is None else int(slc[0].stop - slc[0].start)
        nvox = int(counts[lab_id])

        r_um = eq_radius_um_from_component(nvox, z_extent, voxel_area_xy_um2, voxel_vol_um3)
        r_list.append(r_um)

        cls = int(np.searchsorted(bins_edges, r_um, side="right") - 1)
        cls = max(0, min(n_classes - 1, cls))
        label_to_class[lab_id] = cls

    r_list = np.asarray(r_list, dtype=float)
    if r_list.size > 0:
        print(f"[DEBUG] r_eq ({SIZE_METHOD}) min={np.nanmin(r_list):.3f} um  max={np.nanmax(r_list):.3f} um")

    cls_map = label_to_class[lbl]
    valid_map = is_valid[lbl]

    vols = []
    class_vox_counts = []
    for cls_idx in range(n_classes):
        vox = (cls_map == cls_idx) & valid_map
        nvox_cls = int(vox.sum())
        vols.append(vox.astype(np.uint8))
        class_vox_counts.append(nvox_cls)

    print("[DEBUG] voxels per class:", class_vox_counts)
    print(f"[INFO] Connectivity for visualization: {conn_txt}")

    viewer = napari.Viewer(ndisplay=3)
    viewer.theme = NAPARI_THEME
    viewer.text_overlay.visible = True
    viewer.axes.visible = SHOW_AXES
    viewer.grid.enabled = SHOW_GRID
    viewer.scale_bar.visible = SHOW_SCALE_BAR
    viewer.scale_bar.unit = "µm"
    viewer.scale_bar.position = "bottom_right"

    overlay_text = (
        f"Length: {length_mm:.2f} mm\n"
        f"Diameter: {diameter_mm:.2f} mm\n"
        f"Average porosity: {100.0 * phi_preview_total:.2f}%\n"
        f"Connectivity: {conn_txt}"
    )
    viewer.text_overlay.text = overlay_text
    viewer.text_overlay.color = "black"
    viewer.text_overlay.font_size = 14
    viewer.text_overlay.position = "top_left"

    if SHOW_CORE_SURFACE:
        try:
            core = np.repeat(core2d[np.newaxis, :, :], zp, axis=0).astype(np.uint8)
            verts, faces, _, _ = marching_cubes(core, level=MC_LEVEL, spacing=scale_um)
            add_surface_solid(viewer, verts, faces, CORE_RGB, "Core", CORE_OPACITY)
        except Exception as e:
            print("WARNING: core surface failed:", e)

    fixed_labels = ["0-20 µm", "20-40 µm", "40-80 µm", "80-120 µm", "120-500 µm"]

    for cls_idx in range(n_classes):
        if vols[cls_idx].sum() == 0:
            continue

        color = class_colors[cls_idx]
        name = fixed_labels[cls_idx]

        try:
            verts, faces, _, _ = marching_cubes(vols[cls_idx], level=MC_LEVEL, spacing=scale_um)
            if faces.shape[0] > MAX_FACES_PER_LAYER:
                print(f"WARNING mesh large (fallback to image): {name} faces={faces.shape[0]:,}")
                add_binary_image_solid(viewer, vols[cls_idx], color, name, scale_um)
                continue
            add_surface_solid(viewer, verts, faces, color, name, SURFACE_OPACITY)
        except Exception as e:
            print(f"WARNING: marching_cubes failed (fallback to image): {name} -> {e}")
            add_binary_image_solid(viewer, vols[cls_idx], color, name, scale_um)

    try:
        napari.run()
    except Exception as e:
        raise RuntimeError(
            "Napari could not open because Qt bindings are missing or broken.\n"
            "Install with:\n"
            "python -m pip install pyqt5"
        ) from e


if __name__ == "__main__":
    main()