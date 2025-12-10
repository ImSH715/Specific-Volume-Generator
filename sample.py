import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import SimpleITK as sitk
from Directory_Functions import load_dir

def remove_large_thickness(df):
    df["Images"] = df["Images"].apply(lambda imgs: [f for f in imgs if "_3.0_" not in f])
    return df

def get_kernel_patterns(kernel):
    return [f"0.75_{kernel}_INSPIRATION", f"0.75_{kernel}_EXPIRATION"]

def load_image_array(image_path):
    img = sitk.ReadImage(image_path)
    return sitk.GetArrayFromImage(img)

def load_kernel_images(patient_path, kernel):
    patterns = get_kernel_patterns(kernel)
    try:
        df = load_dir(
            source_dir=patient_path,
            folder="Images",
            timepoint="CT1",
            subfolder=False,
            filenames=patterns
        )
        if df is None or df.empty:
            print(f"[{kernel}] No matching images found.")
            return None
        return remove_large_thickness(df)
    except Exception as e:
        print(f"[{kernel}] load_dir failed for {patient_path} - {e}")
        return None

def load_mask_table(patient_path, patient_id):
    try:
        df = load_dir(
            source_dir=patient_path,
            folder="Masks",
            timepoint="CT1",
            subfolder=False,
            filenames=["INSP", "EXP"]
        )
        if df is None or df.empty:
            print(f"No masks found for {patient_id}, skipping mask application.")
            return None
        return remove_large_thickness(df)
    except Exception as e:
        print(f"[MASKS] load_dir failed for {patient_path} - {e}")
        print(f"No masks found for {patient_id}, skipping mask application.")
        return None

def apply_mask_if_available(image_array, mask_df, breath):
    if mask_df is None:
        return image_array.flatten()

    for _, row in mask_df.iterrows():
        for mname in row["Images"]:
            if breath[:4] in mname:
                mask_path = os.path.join(row["Directory"], mname)
                try:
                    mask_img = sitk.ReadImage(mask_path)
                    mask_arr = sitk.GetArrayFromImage(mask_img)
                    if mask_arr.shape == image_array.shape:
                        return image_array[mask_arr > 0]
                except Exception as e:
                    print(f"Error reading mask: {e}")
    return image_array.flatten()

def plot_histograms(arrays, labels, patient_id, breath):
    plt.figure(figsize=(10, 6))
    for arr, label in zip(arrays, labels):
        plt.hist(arr, bins=100, alpha=0.5, label=label, density=True)
    output_file = f"{patient_id}_{breath}.png"
    plt.title(f"{patient_id} - {breath}")
    plt.xlabel("Intensity")
    plt.ylabel("Frequency")
    plt.legend()
    plt.tight_layout()
    plt.show()
    plt.savefig(output_file)
    plt.close()


def process_patient(patient_id, base_dir):
    print(f"\n=== Processing Patient: {patient_id} ===")
    patient_path = os.path.join(base_dir, patient_id)

    kernels = ["B30f", "B35f", "B60f", "B70f"]
    breaths = ["INSPIRATION", "EXPIRATION"]

    kernel_tables = {k: load_kernel_images(patient_path, k) for k in kernels}
    mask_df = load_mask_table(patient_path, patient_id)

    for breath in breaths:
        arrays_per_kernel = []
        labels = []

        for kernel, df in kernel_tables.items():
            if df is None:
                continue

            all_pixels = []
            for _, row in df.iterrows():
                for fname in row["Images"]:
                    if breath not in fname:
                        continue
                    img_path = os.path.join(row["Directory"], fname)
                    try:
                        img_arr = load_image_array(img_path)
                        masked_arr = apply_mask_if_available(img_arr, mask_df, breath)
                        all_pixels.extend(masked_arr.tolist())
                    except Exception as e:
                        print(f"Error reading image {img_path}: {e}")

            if all_pixels:
                print(f"[{kernel} - {breath}] Loaded {len(all_pixels)} pixels")
                arrays_per_kernel.append(all_pixels)
                labels.append(kernel)
            else:
                print(f"[{kernel} - {breath}] No valid pixels found.")

        if arrays_per_kernel:
            plot_histograms(arrays_per_kernel, labels, patient_id, breath)
        else:
            print(f"No valid data to plot for {patient_id} - {breath}")

def main():
    base_dir = "Z:/polaris2/datasets/NOCTIL_L_CT/cases"
    patient_ids = [pid for pid in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, pid))]

    for patient_id in patient_ids:
        process_patient(patient_id, base_dir)

if __name__ == "__main__":
    main()
