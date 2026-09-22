# -*- coding: utf-8 -*-
"""
Step 42 — Figure 2: Feature evidence drift heatmap
Key features × tasks × target site, |SHAP| ratio (target/source), full-reference
TIFF ≥300 DPI, colorblind-safe sequential palette, publication-ready
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

OUT = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)
R3 = r"E:\ARMD\results\phase3"

# --- data ---
fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
# Pick features with meaningful drift signal (exclude proc/* and adi)
KEY_FEATURES = [
    "comorb_count", "prior_pa_res_levofloxacin",
    "abx_carbapenem_0_30", "abx_carbapenem_181_365",
    "abx_antipseudomonal_bl_0_30", "abx_antipseudomonal_bl_gt365",
    "prior_pa_res_piperacillin_tazobactam", "prior_pa_res_tobramycin",
    "abx_glycopeptide_0_30", "abx_glycopeptide_gt365",
    "prior_pseudo", "prior_pseudo_days",
    "age_bin", "comorb_malignancy", "nh_30d",
]
FEAT_LABELS = {
    "comorb_count": "Comorbidity count",
    "prior_pa_res_levofloxacin": "Prior LVX resistance",
    "abx_carbapenem_0_30": "Carbapenem exp. 0–30d",
    "abx_carbapenem_181_365": "Carbapenem exp. 181–365d",
    "abx_antipseudomonal_bl_0_30": "Anti-PA β-lactam 0–30d",
    "abx_antipseudomonal_bl_gt365": "Anti-PA β-lactam >365d",
    "prior_pa_res_piperacillin_tazobactam": "Prior TZP resistance",
    "prior_pa_res_tobramycin": "Prior TOB resistance",
    "abx_glycopeptide_0_30": "Glycopeptide exp. 0–30d",
    "abx_glycopeptide_gt365": "Glycopeptide exp. >365d",
    "prior_pseudo": "Prior Pseudomonas",
    "prior_pseudo_days": "Days since last Pseudo",
    "age_bin": "Age",
    "comorb_malignancy": "Malignancy",
    "nh_30d": "Nursing home 30d",
}
TASKS = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
DIRS  = ["Stanford", "UTSW"]
TASK_LABELS = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX",
               "ceftazidime": "CAZ", "cefepime": "FEP"}

# Build matrix: rows=feature×dir, cols=task
# Use ratio for each (task, tgt, feature). Some tasks may be missing for certain feature×dir combos.
matrices = {}
for d in DIRS:
    sub = fd[fd["tgt"] == d]
    m = np.full((len(KEY_FEATURES), len(TASKS)), np.nan)
    for i, f in enumerate(KEY_FEATURES):
        for j, t in enumerate(TASKS):
            r = sub[(sub["feature"] == f) & (sub["task"] == t)]
            if len(r): m[i, j] = r["ratio"].iloc[0]
    matrices[d] = m

# Combine Stanford + UTSW into a single tall matrix: (2*N_feat) × N_task
tall = np.vstack([matrices["Stanford"], matrices["UTSW"]])

# --- plotting ---
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8.5})
fig, ax = plt.subplots(figsize=(7.0, 8.0))

# Colorblind-safe sequential: single-hue blue (linear luminance)
from matplotlib.colors import LinearSegmentedColormap
cb_cmap = LinearSegmentedColormap.from_list("cb_blue", ["#f7fbff", "#2171b5"], N=128)
# Diverging around 1.0
cmap = plt.cm.RdBu_r

vmin, vmax = 0.0, 2.0
im = ax.imshow(tall, aspect="auto", cmap=cmap, vmin=vmin, vmax=vmax)

# Center at 1.0 for diverging: use two-segment norm
import matplotlib.colors as mcolors
norm = mcolors.TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=vmax)
im = ax.imshow(tall, aspect="auto", cmap=cmap, norm=norm)

# Labels
n_feat = len(KEY_FEATURES)
ax.set_yticks(range(2 * n_feat))
ax.set_yticklabels([FEAT_LABELS.get(f, f) for f in KEY_FEATURES] * 2, fontsize=7.5)
ax.set_xticks(range(len(TASKS)))
ax.set_xticklabels([TASK_LABELS[t] for t in TASKS], fontsize=8.5, weight="bold")

# Divider line between Stanford and UTSW
ax.axhline(y=n_feat - 0.5, color="black", linewidth=0.8)

# Site labels
ax.text(-1.0, n_feat/2 - 0.5, "Stanford", rotation=90, va="center", ha="center", fontsize=9,
        weight="bold")
ax.text(-1.0, n_feat + n_feat/2 - 0.5, "UTSW", rotation=90, va="center", ha="center", fontsize=9,
        weight="bold")

# Annotate cells
for i in range(tall.shape[0]):
    for j in range(tall.shape[1]):
        v = tall[i, j]
        if not np.isnan(v):
            color = "white" if (v < 0.4 or v > 1.6) else "black"
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5, color=color)

# Colorbar
cbar = fig.colorbar(im, ax=ax, shrink=0.82, pad=0.02)
cbar.set_label("|SHAP| ratio (target / source)", fontsize=8.5)
cbar.ax.axhline(y=1.0, color="black", linewidth=0.7, linestyle="--")

ax.set_title("Figure 2. Feature evidence drift: |SHAP| ratio (target / source)",
             fontsize=10.5, weight="bold", pad=10)
fig.text(0.5, 0.01, "Ratio < 1 = evidence attenuated after transfer. Ratio > 1 = evidence amplified. "
         "Same MGB-trained model, full-reference SHAP. See Methods §7.1.",
         ha="center", fontsize=7, style="italic")

plt.tight_layout(rect=[0, 0.03, 1, 0.97])
fig.savefig(os.path.join(OUT, "Figure2_feature_drift_heatmap.tif"), dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 2 saved:", os.path.join(OUT, "Figure2_feature_drift_heatmap.tif"))
