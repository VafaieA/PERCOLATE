# PERCOLATE-V1
Pore Extraction and Reconstruction for COre analysis and AnalyTical Evaluation
# PERCOLATE

### Pore Extraction and Reconstruction for COre anaLysis using micro-CT

PERCOLATE is a reproducible micro-CT analysis framework designed for **cylindrical core samples**, enabling pore segmentation, porosity quantification, pore size distribution (PSD) analysis, and physically accurate 3D visualization.

---

## ✨ Features

* Interactive segmentation parameter tuning
* Automated batch pore segmentation
* Binary pore mask generation (TIFF stack)
* Porosity computation (full volume)
* Equivalent pore radius distribution (PSD)
* 3D connected-component pore analysis
* Physically scaled 3D visualization in Napari
* Fully reproducible workflow via JSON parameter tracking

---

## 🧭 Workflow

PERCOLATE consists of four main steps:

### 1. Interactive Segmentation

```bash
python 01-Interactive-segmentation.py
```

* Define crop region
* Define cylindrical core mask
* Adjust threshold (manual / Otsu)
* Save parameters to JSON

---

### 2. Batch Segmentation

```bash
python 02A-Batch segmentation.py
```

* Applies saved parameters to all slices
* Generates binary pore mask stack

---

### 3. Pore Clustering & PSD

```bash
python 02B-Pore-Clustering.py
```

* Computes total porosity
* Performs 3D connected-component labeling
* Calculates equivalent pore radius
* Outputs PSD and statistics

---

### 4. 3D Visualization

```bash
python 03-Visulaizer-Pore-Clustering.py
```

* Loads pore volume (preview or full)
* Applies physical scaling (µm)
* Classifies pores into size bins
* Interactive visualization in Napari

---

## 📁 Example Dataset

A **lightweight example dataset (30 slices)** is included for demonstration and reproducibility.

* Format: TIFF stack
* Naming convention:

```text
block00000019_z0000.tif
block00000019_z0001.tif
...
```

Where:

* `block00000019` → sample identifier
* `zXXXX` → slice index

⚠️ Note:
The dataset is intentionally limited to 30 slices due to GitHub storage constraints. Full datasets may contain significantly more slices.

---

## 📦 Outputs

* `mask_stack.tif` → binary pore volume
* `pore_preview_3d.tif` → downsampled 3D pore volume
* `pore_eq_radius_hist_um.csv` → PSD histogram
* `pore_components.csv` → component statistics
* `summary.json` → porosity and metadata

---

## 📐 Pore Size Definition

Equivalent pore radius is computed as:

[
r_{eq} = \sqrt{\frac{A}{\pi}}
]

Two modes are available:

* `xy_area_est` (default)
* `sphere_volume`

---

## ⚙️ Requirements

* Python ≥ 3.9
* numpy
* scipy
* scikit-image
* matplotlib
* tifffile
* napari

Install:

```bash
pip install numpy scipy scikit-image matplotlib tifffile napari pyqt5
```

---

## 📊 Input Data

* 3D grayscale TIFF stack
* Cylindrical core geometry
* Typical voxel size: ~7.14 µm

---

## ⚠️ Important Notes

* PSD represents **CT-resolved pore clusters**, not pore throats
* Results are **resolution-dependent**
* Suitable for **comparative analysis between samples**

---

## 📁 Repository Structure

```
PERCOLATE/
│
├── 01-Interactive-segmentation.py
├── 02A-Batch segmentation.py
├── 02B-Pore-Clustering.py
├── 03-Visulaizer-Pore-Clustering.py
│
├── Raw_TIFF_Slices/
├── Segmentation_Parameters/
├── Batch_Output/
├── Pore_Clustering_Output/
│
└── README.md
```

---

## 📌 Citation

If you use PERCOLATE in your research, please cite:

**PERCOLATE: A reproducible micro-CT framework for cylindrical core pore segmentation and 3D quantitative analysis**
URL: 
---

## 👨‍🔬 Author

Atefeh Vafaie, Iman R. Kivi, Victor Vilarrasa
Global Change Research Group, IMEDEA, CSIC-UIB, Esporles, Spain
[Your Institution]
[Your Email]
