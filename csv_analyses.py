import os
import re
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

CSV_PATH = "SVg_mean.csv"
OUT_SUMMARY = "mean_svg_by_kernel_phase_group.csv"

OUT_DIR = "charts_kernel_phase"
OUT_DIR_COMBINED = "charts_combined"
OUT_DIR_DISEASE = "charts_by_disease"

os.makedirs(OUT_DIR_DISEASE, exist_ok=True)
os.makedirs(OUT_DIR_COMBINED, exist_ok=True)
os.makedirs(OUT_DIR, exist_ok=True)

df = pd.read_csv(CSV_PATH)

# Normalize / validate columns coming from the spreadsheet
REQUIRED = ["disease", "phase", "kernel", "SVg_mean"]
missing = [c for c in REQUIRED if c not in df.columns]
if missing:
    raise ValueError(f"CSV is missing required columns: {missing}")

# Clean up strings and ensure consistent categories
def norm_kernel(k: str) -> str:
    k = str(k).strip().upper()
    m = re.fullmatch(r"B(\d{2})F?", k, flags=re.IGNORECASE)
    if m:
        return f"B{m.group(1)}F"
    return "UNK"

def norm_phase(p: str) -> str:
    p = str(p).strip().upper()
    if p.startswith("INSP"):
        return "INSPIRATION"
    if p.startswith("EXP"):
        return "EXPIRATION"
    return "UNKNOWN"

def norm_disease(d: str) -> str:
    d = str(d).strip().capitalize()
    if d.lower().startswith("asth"):
        return "Asthma"
    if d.lower().startswith("copd") or d.upper() in {"MRC", "LEIC"}:
        return "COPD"
    if d.lower().startswith("heal") or d.upper().startswith("N"):
        return "Healthy"
    return "Unknown"

df["kernel"]  = df["kernel"].apply(norm_kernel)
df["phase"]   = df["phase"].apply(norm_phase)
df["disease"] = df["disease"].apply(norm_disease)

# Filter to valid rows only
df = df[
    (df["kernel"].str.match(r"^B\d{2}F$")) &
    (df["phase"].isin(["INSPIRATION", "EXPIRATION"])) &
    (df["disease"].isin(["Asthma", "COPD", "Healthy"]))
].copy()

# Summarize means
def summarize_means(sub: pd.DataFrame) -> pd.DataFrame:
    if sub.empty:
        return pd.DataFrame(columns=["kernel","phase","disease","mean_SVg_mL_per_g","n"])
    g = sub.groupby(["kernel","phase","disease"])["SVg_mean"].agg(["mean","count"]).reset_index()
    g = g.rename(columns={"mean":"mean_SVg_mL_per_g","count":"n"})
    kernel_order = ["B30F","B35F","B60F","B70F"]
    disease_order = ["Asthma","COPD","Healthy"]
    phase_order   = ["INSPIRATION","EXPIRATION"]
    g["kernel"]  = pd.Categorical(g["kernel"],  categories=kernel_order, ordered=True)
    g["disease"] = pd.Categorical(g["disease"], categories=disease_order, ordered=True)
    g["phase"]   = pd.Categorical(g["phase"],   categories=phase_order,   ordered=True)
    return g.sort_values(["kernel","phase","disease"]).reset_index(drop=True)

summary_tbl = summarize_means(df)
summary_tbl.to_csv(OUT_SUMMARY, index=False)

# Plotting
def plot_kernel_phase_bars(table: pd.DataFrame, out_dir: str):
    if table.empty:
        return
    kernels  = table["kernel"].cat.categories
    diseases = table["disease"].cat.categories
    phases   = table["phase"].cat.categories

    for kernel in kernels:
        tbl_k = table[table["kernel"] == kernel]
        if tbl_k.empty:
            continue

        x = np.arange(len(phases))
        width = 0.8 / len(diseases)

        fig, ax = plt.subplots(figsize=(8,6))
        for i, disease in enumerate(diseases):
            vals = []
            for phase in phases:
                row = tbl_k[(tbl_k["phase"]==phase) & (tbl_k["disease"]==disease)]
                vals.append(float(row["mean_SVg_mL_per_g"].iloc[0]) if not row.empty else np.nan)
            positions = x - 0.4 + width/2 + i*width
            ax.bar(positions, vals, width=width, label=str(disease))
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x, phases)
        ax.set_ylabel("Mean SVg (mL/g)")
        ax.set_title(f"{kernel} : Mean SVg by Phase and Disease")
        ax.legend()
        fig.tight_layout()
        out_path = os.path.join(out_dir, f"{kernel}_phase_group_bar.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)

def plot_all_kernels_phases(table: pd.DataFrame, out_path: str):
    if table.empty:
        return
    kernels  = table["kernel"].cat.categories
    phases   = table["phase"].cat.categories
    diseases = table["disease"].cat.categories

    x_labels = [f"{k}-{p}" for k in kernels for p in phases]
    x = np.arange(len(x_labels))
    width = 0.8 / len(diseases)

    fig, ax = plt.subplots(figsize=(12,6))
    for i, disease in enumerate(diseases):
        vals = []
        for k in kernels:
            for p in phases:
                row = table[(table["kernel"]==k) & (table["phase"]==p) & (table["disease"]==disease)]
                vals.append(float(row["mean_SVg_mL_per_g"].iloc[0]) if not row.empty else np.nan)
        positions = x - 0.4 + width/2 + i*width
        ax.bar(positions, vals, width=width, label=str(disease))
        for xpos, val in zip(positions, vals):
            if not np.isnan(val):
                ax.text(xpos, val, f"{val:.1f}", ha="center", va="bottom", fontsize=7)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels, rotation=45, ha="right")
    ax.set_ylabel("Mean SVg (mL/g)")
    ax.set_title("Mean SVg by Kernel-Phase and Disease")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] All kernels & phases chart saved → {out_path}")

def plot_bars_by_disease(table, out_dir=OUT_DIR_DISEASE):
    if table is None or table.empty:
        print("[WARN] No data to plot for disease-based charts.")
        return

    # Ensure consistent category order if available
    if "disease" in table.columns and hasattr(table["disease"], "cat"):
        diseases = list(table["disease"].cat.categories)
    else:
        diseases = sorted(table["disease"].unique())

    if "kernel" in table.columns and hasattr(table["kernel"], "cat"):
        kernels = list(table["kernel"].cat.categories)
    else:
        kernels = sorted(table["kernel"].unique())

    if "phase" in table.columns and hasattr(table["phase"], "cat"):
        phases = list(table["phase"].cat.categories)
    else:
        phases = sorted(table["phase"].unique())

    for disease in diseases:
        tbl_d = table[table["disease"] == disease]
        if tbl_d.empty:
            print(f"[INFO] Skip {disease}: no rows.")
            continue

        # Prepare grouped bars: each kernel has |phases| bars
        x = np.arange(len(kernels))
        width = 0.8 / max(1, len(phases))

        fig, ax = plt.subplots(figsize=(10, 6))
        for i, phase in enumerate(phases):
            # Collect values per kernel for this phase
            vals = []
            for k in kernels:
                row = tbl_d[(tbl_d["kernel"] == k) & (tbl_d["phase"] == phase)]
                if row.empty:
                    vals.append(np.nan)
                else:
                    vals.append(float(row["mean_SVg_mL_per_g"].iloc[0]))

            # Bar positions for this phase within each kernel group
            positions = x - 0.4 + width/2 + i*width
            ax.bar(positions, vals, width=width, label=str(phase))

            # Optional data labels
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x, kernels)
        ax.set_ylabel("Mean SVg (mL/g)")
        ax.set_title(f"Mean SVg by Kernel and Phase — {disease}")
        ax.legend(title="Phase")
        fig.tight_layout()

        out_path = os.path.join(out_dir, f"{disease}_by_kernel_phase.png")
        fig.savefig(out_path, dpi=150)
        plt.close(fig)
        print(f"[OK] Saved disease-based chart → {out_path}")

def plot_all_diseases_combined(table, out_path=os.path.join(OUT_DIR_COMBINED, "all_diseases_combined.png")):
    if table is None or table.empty:
        print("[WARN] No data to plot for combined chart.")
        return

    # Resolve category orders if provided
    if "phase" in table.columns and hasattr(table["phase"], "cat"):
        phases = list(table["phase"].cat.categories)
    else:
        phases = sorted(table["phase"].unique())

    if "kernel" in table.columns and hasattr(table["kernel"], "cat"):
        kernels = list(table["kernel"].cat.categories)
    else:
        kernels = sorted(table["kernel"].unique())

    if "disease" in table.columns and hasattr(table["disease"], "cat"):
        diseases = list(table["disease"].cat.categories)
    else:
        diseases = sorted(table["disease"].unique())

    # Ensure exactly two panels if those phases exist
    if len(phases) == 0:
        print("[WARN] No phases found.")
        return

    n_panels = len(phases)
    fig, axes = plt.subplots(1, n_panels, figsize=(10 if n_panels == 1 else 16, 6), sharey=True)
    if n_panels == 1:
        axes = [axes]

    x = np.arange(len(kernels))
    width = 0.8 / max(1, len(diseases))

    for ax, phase in zip(axes, phases):
        tbl_p = table[table["phase"] == phase]
        if tbl_p.empty:
            ax.set_title(f"{phase} (no data)")
            continue

        # Bars by disease within each kernel
        for i, disease in enumerate(diseases):
            vals = []
            for k in kernels:
                row = tbl_p[(tbl_p["kernel"] == k) & (tbl_p["disease"] == disease)]
                vals.append(float(row["mean_SVg_mL_per_g"].iloc[0]) if not row.empty else np.nan)

            positions = x - 0.4 + width/2 + i*width
            ax.bar(positions, vals, width=width, label=str(disease))

            # Optional value labels
            for xpos, val in zip(positions, vals):
                if not np.isnan(val):
                    ax.text(xpos, val, f"{val:.1f}", ha="center", va="bottom", fontsize=8)

        ax.set_xticks(x, kernels)
        ax.set_title(f"{phase}")
        ax.set_xlabel("Kernel")

    axes[0].set_ylabel("Mean SVg (mL/g)")
    # Put legend once, on the last axis
    handles, labels = axes[-1].get_legend_handles_labels()
    axes[-1].legend(handles, labels, title="Disease", loc="best")

    fig.suptitle("Mean SVg by Kernel — All Diseases (split by Phase)")
    fig.tight_layout(rect=[0, 0.03, 1, 0.95])
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"[OK] Saved combined chart → {out_path}")

# Run plots
plot_kernel_phase_bars(summary_tbl, OUT_DIR)
plot_all_kernels_phases(summary_tbl, os.path.join(OUT_DIR, "all_kernels_phases.png"))
plot_bars_by_disease(summary_tbl, OUT_DIR_DISEASE)
plot_all_diseases_combined(summary_tbl)

print(f"[OK] Summary saved: {OUT_SUMMARY}")
print(f"[OK] Charts saved in: {OUT_DIR}")
