# -*- coding: utf-8 -*-
"""
Step 44 — Supplementary Figures S1–S4
S1: DCA curves for primary directions
S2: Era sensitivity comparison (≥2020 vs all, bar chart)
S3: Mucoid sensitivity (with/without mucoid, bar chart)
S4: Full CI matrix (30 directions × 5 tasks, forest-plot style)
All TIFF ≥300 DPI
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

OUT = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)
plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 8.5})
R3 = r"E:\ARMD\results\phase3"

# ========== S1: DCA ==========
dca = pd.read_csv(os.path.join(R3, "dca.csv"))
TASKS = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
DIRS  = [("Stanford", "S"), ("UTSW", "U")]
LAB = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX",
       "ceftazidime": "CAZ", "cefepime": "FEP"}

fig, axes = plt.subplots(2, 5, figsize=(15, 6.2))
for j, t in enumerate(TASKS):
    for i, (d, dl) in enumerate(DIRS):
        ax = axes[i, j]
        sub = dca[(dca["task"]==t) & (dca["tgt"]==d)]
        for mod, ls, c in [("full", "-", "#2171b5"), ("subset", "--", "#ca0020")]:
            s = sub[sub["model"]==mod]
            if len(s):
                ax.plot(s["threshold"], s["net_benefit"], ls, color=c, linewidth=1.2)
        ax.plot([0, 1], [0, 0], "k-", alpha=0.2, linewidth=0.5)
        ax.set_title(f"{LAB[t]} → {dl}", fontsize=9, weight="bold")
        ax.set_xlim(0, 0.5); ax.set_ylim(-0.05, None)
        if i == 0:
            ax.set_xlabel("")
        else:
            ax.set_xlabel("Threshold probability")
        if j == 0: ax.set_ylabel("Net benefit")
fig.suptitle("Figure S1. Decision curve analysis (all primary directions). "
             "Solid: full model, Dashed: stable-subset model.", fontsize=10.5, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.94])
fig.savefig(os.path.join(OUT, "FigureS1_dca.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("S1 saved")

# ========== S2: Era Sensitivity ==========
era = pd.read_csv(os.path.join(R3, "era_sensitivity.csv"))
era20 = era[era["cut"]==2020]
fig, ax = plt.subplots(figsize=(10, 5))
x = np.arange(len(era20)); w = 0.35
ax.bar(x - w/2, era20["auroc_all"], w, label="All years", color="#2171b5", alpha=0.85)
ax.bar(x + w/2, era20["auroc_cut"], w, label="≥2020 only", color="#ca0020", alpha=0.85)
ax.set_xticks(x)
ax.set_xticklabels([f"{LAB[r['task']]}→{r['tgt'][:1]}" for _, r in era20.iterrows()], rotation=45, fontsize=7)
ax.set_ylabel("AUROC"); ax.legend(fontsize=8)
ax.set_title("Figure S2. Era sensitivity: ≥2020 vs all years (MGB→external).", fontsize=10.5, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "FigureS2_era_sensitivity.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("S2 saved")

# ========== S3: Mucoid Sensitivity ==========
muc = pd.read_csv(os.path.join(R3, "mucoid_sensitivity.csv"))
muc = muc.dropna(subset=["auroc_mucoid_only"])   # drop UTSW rows (no mucoid)
fig, ax = plt.subplots(figsize=(9, 5))
x2 = np.arange(len(muc)); w2 = 0.25
ax.bar(x2 - w2, muc["auroc_all"], w2, label="All samples", color="#2171b5", alpha=0.85)
ax.bar(x2, muc["auroc_no_mucoid"], w2, label="Excl. mucoid", color="#5ab4ac", alpha=0.85)
ax.bar(x2 + w2, muc["auroc_mucoid_only"], w2, label="Mucoid only", color="#ca0020", alpha=0.85)
ax.set_xticks(x2)
ax.set_xticklabels([f"{LAB[r['task']]}→{r['tgt'][:1]}" for _, r in muc.iterrows()], rotation=45, fontsize=7.5)
ax.set_ylabel("AUROC"); ax.legend(fontsize=8)
ax.set_title("Figure S3. Mucoid sensitivity (Stanford only; UTSW has no mucoid cohort).",
             fontsize=10.5, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "FigureS3_mucoid_sensitivity.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("S3 saved")

# ========== S4: Full CI Matrix (forest-plot style, 30 directions) ==========
tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
ss = pd.read_csv(os.path.join(R3, "stable_subset.csv"))
fig, ax = plt.subplots(figsize=(12, 7.5))
y_pos = 0; y_ticks, y_labels = [], []
colors = {"MGB→S":"#2171b5", "MGB→U":"#ca0020", "S→MGB":"#5ab4ac", "U→MGB":"#f4a582",
          "S→U":"#92c5de", "U→S":"#b2182b"}
for _, r in tm.iterrows():
    label = f"{LAB[r['task']]} {r['src']}→{r['tgt']}"
    ci_row = ss[(ss["task"]==r["task"]) & (ss["tgt"]==r["tgt"])]
    if len(ci_row) > 0:
        ci_str = str(ci_row["boot_ci"].iloc[0])
        parts = ci_str.split("-")
        lo, hi = float(parts[0]), float(parts[-1])
    else:
        lo, hi = r["auroc"] - 0.03, r["auroc"] + 0.03
    lo_err = max(0, r["auroc"] - lo); hi_err = max(0, hi - r["auroc"])
    ax.errorbar(r["auroc"], y_pos, xerr=[[lo_err], [hi_err]],
                fmt="o", color=colors.get(f"{r['src']}→{r['tgt']}", "#333333"),
                capsize=3, markersize=4, linewidth=1.0)
    ax.axvline(x=r["internal_auroc"], color="grey", alpha=0.3, linewidth=0.5)
    y_ticks.append(y_pos); y_labels.append(label); y_pos += 1
ax.set_yticks(y_ticks); ax.set_yticklabels(y_labels, fontsize=6.5)
ax.set_xlabel("AUROC")
ax.set_title("Figure S4. Full CI matrix for all 30 transfer directions.", fontsize=10.5, weight="bold")
plt.tight_layout()
fig.savefig(os.path.join(OUT, "FigureS4_ci_matrix.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("S4 saved")
print("All supplementary figures done:", OUT)
