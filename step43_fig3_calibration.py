# -*- coding: utf-8 -*-
"""
Step 43 — Figure 3: Calibration curves (MGB→S/U, representative tasks MEM/CAZ/FEP)
TIFF ≥300 DPI, publication-ready
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.calibration import calibration_curve
import os

OUT = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)
import sys; sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross

CLEAN = r"E:\ARMD\data\clean"
TASKS = ["meropenem", "cefepime", "ceftazidime"]       # representative
DIRS  = [("Stanford", "S"), ("UTSW", "U")]
TASK_LABELS = {"meropenem": "MEM", "cefepime": "FEP", "ceftazidime": "CAZ"}

plt.rcParams.update({"font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
                     "font.size": 9})
fig, axes = plt.subplots(2, 3, figsize=(10, 7.2))
axes = axes.flatten()

idx = 0
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    mgb = prep(mgb_raw)
    feats = feature_cols(mgb)
    maps = fit_cat_maps(mgb_raw)
    m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
    for d, dl in DIRS:
        te_raw = pd.read_csv(os.path.join(CLEAN, d, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        te = prep_cross(te_raw, maps, mgb["adi"].median())
        X = te[[f for f in feats if f in te.columns]]
        y = te["label"].values
        pred = m.predict_proba(X)[:, 1]
        frac_pos, mean_pred = calibration_curve(y, pred, n_bins=10, strategy="uniform")
        ax = axes[idx]
        ax.plot([0, 1], [0, 1], "k--", alpha=0.3, linewidth=0.8)
        ax.plot(mean_pred, frac_pos, "o-", color="#2171b5", markersize=4, linewidth=1.3)
        ax.set_title(f"{TASK_LABELS[t]} → {dl}", fontsize=10, weight="bold")
        ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction positive")
        ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
        # Brier from transfer_matrix
        tm = pd.read_csv(os.path.join(r"E:\ARMD\results\phase3", "transfer_matrix.csv"))
        r = tm[(tm["task"]==t) & (tm["src"]=="MGB") & (tm["tgt"]==d)]
        brier = f"{r['brier'].iloc[0]:.3f}" if len(r) else "NA"
        ax.text(0.95, 0.08, f"Brier={brier}", ha="right", va="bottom", transform=ax.transAxes,
                fontsize=8, color="grey")
        idx += 1

fig.suptitle("Figure 3. Calibration curves (MGB→external, representative tasks).",
             fontsize=11, weight="bold")
plt.tight_layout(rect=[0, 0, 1, 0.95])
fig.savefig(os.path.join(OUT, "Figure3_calibration_curves.tif"), dpi=300,
            pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 3 saved")
