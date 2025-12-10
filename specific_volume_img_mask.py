# -*- coding: utf-8 -*-
"""
This script calculates the mean Specific Gas Volume (SVg) for lung CT images.

Processing steps:
1. Load NIfTI images and corresponding masks from the directory structure.
2. For each phase (INSPIRATION, EXPIRATION), select one mask.
3. Apply the same phase-specific mask to all kernels (B30F, B35F, B60F, B70F) in that phase.
4. Convert Hounsfield Units (HU) in the masked region to Specific Gas Volume (SVg).
5. Compute HU statistics (min, max, mean) for debugging.
6. Save results to CSV with detailed metadata.
"""

import os
import re
import glob
import numpy as np
import pandas as pd
import SimpleITK as sitk
from Directory_Functions import load_dir

CASES_ROOT = r"/shared/polaris2/datasets/NOCTIL_L_CT/cases"
TIMEPOINTS = None
FOLDER     = None
SUBFOLDERS = ["Images", "Masks"]
FILENAME_FILTERS = None
OUT_CSV    = "SVg_mean.csv"

SV_TISSUE = 1.0 / 1.065  # mL/g for tissue density
DEBUG = True  # Set to True to print per-case debug information

def hu_to_svg_mean(hu_vals: np.ndarray) -> float:
    """
    Convert HU values to Specific Gas Volume (SVg) and return the mean.
    Filters out extreme HU values that correspond to air/background.
    """
    hu = np.asarray(hu_vals, dtype=np.float16)

    # Remove obvious background (-1024 HU)
    hu = hu[hu > -1020]

    # Optional: filter HU to plausible lung tissue range
    hu = hu[(hu >= -1000) & (hu <= 200)]
    if hu.size == 0:
        return np.nan

    svg = 1024.0 / (hu + 1024.0) - SV_TISSUE
    return float(np.mean(svg))


def sort_phase(name: str) -> str:
    """
    Determine respiratory phase from filename.
    Returns 'INSPIRATION', 'EXPIRATION', or 'UNKNOWN'.
    """
    n = str(name).upper()
    if "INSPIRATION" in n or re.search(r"(^|_)INSP(_|$)", n):
        return "INSPIRATION"
    if "EXPIRATION" in n or re.search(r"(^|_)EXP(_|$)", n):
        return "EXPIRATION"
    return "UNKNOWN"

def sort_kernel(name: str) -> str:
    """
    Extract reconstruction kernel name from filename (e.g., B30F).
    """
    m = re.search(r"(B\d{2}F?)", str(name), flags=re.IGNORECASE)
    if not m:
        return "UNK"
    k = m.group(1).upper()
    return k if k.endswith("F") else (k + "F")

def sort_disease(case_name: str) -> str:
    """
    Determine disease group from case name.
    - Asthma: case names starting with 'A' or '0001'
    - COPD: case names starting with 'LEIC' or 'MRC'
    - Healthy: case names starting with 'N'
    """
    t = str(case_name).upper()
    if t.startswith("A") or t.startswith("0001"):
        return "Asthma"
    if t.startswith("LEIC") or t.startswith("MRC"):
        return "COPD"
    if t.startswith("N"):
        return "Healthy"
    return "Unknown"

def first(glob_list):
    """
    Return the first element from a sorted list, or None if the list is empty.
    """
    return sorted(glob_list)[0] if glob_list else None

def adjust_mask(img_path: str, mask_path: str):
    """
    Read an image and mask, and resample the mask to match the image's size,
    spacing, origin, and direction if necessary.

    Returns:
        img_np: numpy array of the image (float16)
        mask_np: boolean numpy array of the mask
    """
    im = sitk.ReadImage(img_path)
    mk = sitk.ReadImage(mask_path)

    need_resample = (
        im.GetSize()      != mk.GetSize()      or
        im.GetSpacing()   != mk.GetSpacing()   or
        im.GetOrigin()    != mk.GetOrigin()    or
        im.GetDirection() != mk.GetDirection()
    )

    if need_resample:
        res = sitk.ResampleImageFilter()
        res.SetReferenceImage(im)
        res.SetInterpolator(sitk.sitkNearestNeighbor)
        res.SetTransform(sitk.Transform())
        mk = res.Execute(mk)

    img_np  = sitk.GetArrayFromImage(im).astype(np.float16)
    mask_np = sitk.GetArrayFromImage(mk) > 0
    return img_np, mask_np

def find_svg_mean():
    """
    Main function to:
    1. Load directory structure
    2. Match images and masks by case and session
    3. For each phase, select one mask and apply to all kernels
    4. Compute SVg mean and HU statistics
    5. Save results to CSV
    """
    tbl = load_dir(
        source_dir=CASES_ROOT,
        folder=FOLDER,
        subfolder=SUBFOLDERS,
        timepoint=TIMEPOINTS,
        filenames=FILENAME_FILTERS,
    )
    if tbl is None or tbl.empty:
        raise RuntimeError("load_dir() returned empty DataFrame.")

    # Group paths by case|session
    sessions = {}
    for _, row in tbl.iterrows():
        d = row["Directory"]
        if not isinstance(d, str):
            continue
        parent = os.path.dirname(d)
        case   = os.path.basename(os.path.dirname(parent))
        session= os.path.basename(parent)
        leaf   = os.path.basename(d)
        key = f"{case}|{session}"
        sessions.setdefault(key, {})
        if leaf.lower() == "images":
            sessions[key]["Images"] = d
        elif leaf.lower() == "masks":
            sessions[key]["Masks"] = d

    rows = []
    for key, paths in sessions.items():
        case, session = key.split("|")
        img_dir  = paths.get("Images")
        mask_dir = paths.get("Masks")
        if not img_dir or not mask_dir:
            continue

        img_files = sorted(glob.glob(os.path.join(img_dir, "*.nii"))) + \
                    sorted(glob.glob(os.path.join(img_dir, "*.nii.gz")))
        mask_files = sorted(glob.glob(os.path.join(mask_dir, "*.nii"))) + \
                     sorted(glob.glob(os.path.join(mask_dir, "*.nii.gz")))
        if not img_files or not mask_files:
            continue

        # Select one mask per phase
        masks_by_phase = {"INSPIRATION": None, "EXPIRATION": None}
        for phase in masks_by_phase.keys():
            cands = [p for p in mask_files if sort_phase(p) == phase]
            pick = first([p for p in cands if "lobar_seg_checked" in os.path.basename(p)]) \
                or first([p for p in cands if "seg" in os.path.basename(p)]) \
                or first([p for p in cands if "mask" in os.path.basename(p)]) \
                or first(cands)
            masks_by_phase[phase] = pick

        # Apply the same mask to all kernels for each phase
        for ipath in img_files:
            phase = sort_phase(ipath)
            kernel = sort_kernel(ipath)
            mask_path = masks_by_phase.get(phase)
            if not mask_path:
                continue
            try:
                img_np, mask_np = adjust_mask(ipath, mask_path)
                if not np.any(mask_np):
                    continue

                hu_vals = img_np[mask_np]
                mean_svg = hu_to_svg_mean(hu_vals)

                # Compute HU statistics
                hu_min, hu_max, hu_mean = float(np.min(hu_vals)), float(np.max(hu_vals)), float(np.mean(hu_vals))
                voxels = int(mask_np.sum())

                # Debug output
                if DEBUG:
                    print(f"[DEBUG] {case} | {phase} | {kernel} | voxels={voxels} "
                          f"| HU(min={hu_min:.1f}, max={hu_max:.1f}, mean={hu_mean:.1f}) "
                          f"| SVg={mean_svg:.2f}")

                # Abnormal value warning
                if abs(mean_svg) > 100:
                    print(f"[WARNING] Abnormal SVg ({mean_svg:.2f}) detected in {case}, {phase}, {kernel}")

                rows.append({
                    "case": case,
                    "session": session,
                    "disease": sort_disease(case),
                    "phase": phase,
                    "kernel": kernel,
                    "SVg_mean": mean_svg,
                    "voxels": voxels,
                    "HU_min": hu_min,
                    "HU_max": hu_max,
                    "HU_mean": hu_mean,
                    "image": os.path.basename(ipath),
                    "mask": os.path.basename(mask_path)
                })
            except Exception as e:
                print(f"[ERROR] Failed: {ipath} with {mask_path} ({e})")
                continue

    # Save results
    df = pd.DataFrame(rows)
    df = df[["case","session","disease","phase","kernel","SVg_mean",
             "voxels","HU_min","HU_max","HU_mean","image","mask"]]
    df.to_csv(OUT_CSV, index=False)
    print(f"[OK] Saved: {os.path.abspath(OUT_CSV)}  (rows={len(df)})")
    return df

if __name__ == "__main__":
    find_svg_mean()
