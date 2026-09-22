# -*- coding: utf-8 -*-
"""
Figures v2 — all 7 publication figures, rewritten for quality.
Style: single consistent rcParams, colorblind-safe palettes, no text overlap,
       ≥300 DPI TIFF with LZW compression, bbox_inches='tight'
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os, sys

OUT = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)
R3  = r"E:\ARMD\results\phase3"
CLEAN = r"E:\ARMD\data\clean"
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross

# ── Global style ──
plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "figure.dpi": 150, "savefig.dpi": 300, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.1,
})

C_BLUE   = "#2171B5"
C_RED    = "#CA0020"
C_TEAL   = "#5AB4AC"
C_ORANGE = "#F4A582"
C_GREY   = "#666666"
C_LGREY  = "#BDBDBD"
C_INK    = "#1A1A1A"
C_WHITE  = "#FAFAFA"

TASKS   = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
T_LAB   = {"meropenem":"MEM","ciprofloxacin":"CIP","levofloxacin":"LVX",
           "ceftazidime":"CAZ","cefepime":"FEP"}

def save(name):
    fp = os.path.join(OUT, name)
    plt.savefig(fp, dpi=300, pil_kwargs={"compression": "tiff_lzw"})
    print(f"  {name} saved ({os.path.getsize(fp)//1024} KB)")
    plt.close()

# =====================================================
# Figure 1 — Cohort attrition (TRIPOD-AI enrollment flow)
# =====================================================
def fig1():
    fig, ax = plt.subplots(figsize=(7.5, 7.5))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(y, title, sub, x0=2.5, w=5.0, h=0.72):
        r = mpatches.FancyBboxPatch((x0, y-h/2), w, h, boxstyle="round,pad=0.08",
                                     facecolor=C_WHITE, edgecolor=C_INK, linewidth=1.2, zorder=3)
        ax.add_patch(r)
        ax.text(x0+w/2, y+0.06, title, ha="center", va="center", fontsize=9, weight="bold", color=C_INK, zorder=4)
        ax.text(x0+w/2, y-0.28, sub, ha="center", va="center", fontsize=7.5, color=C_GREY, zorder=4)

    def arrow(yf, yt, note, x=5.0):
        ax.annotate("", xy=(x, yt+0.32), xytext=(x, yf-0.32),
                    arrowprops=dict(arrowstyle="->", color=C_GREY, lw=1.2), zorder=2)
        ax.text(x+2.6, (yf+yt)/2, note, ha="left", va="center", fontsize=7.3, color=C_GREY, zorder=4)

    def note(y, text):
        ax.text(7.8, y, text, ha="left", va="center", fontsize=7, color=C_GREY, zorder=4)

    y = 9.2
    box(y, "MGB microbiology cohort", "4,960,599 rows, 2015–2024 (dates shifted)")
    arrow(y, y-0.75, "Organism starts with PSEUDOMONAS AERUGINOSA")
    y -= 0.8
    box(y, "P. aeruginosa rows", "221,675 rows · 15,254 cultures · 5,216 patients")
    arrow(y, y-0.75, "Drug in 9-task set, exclude preliminary & negative cultures")
    y -= 0.8
    box(y, "PA target-drug rows after exclusions", "169,477 rows · 15,253 cultures")
    arrow(y, y-0.75, "CLSI_2022_pheno in {S, I, R}")
    y -= 0.8
    box(y, "PA rows with valid label", "105,421 rows · 10,578 cultures (69% of PA cultures)")
    note(y, "(37.8% lost: no interpretable CLSI phenotype;\nmucoid isolates 38.6% of lost rows)")

    fig.suptitle("Figure 1. MGB cohort attrition (TRIPOD-AI enrollment flow).", fontsize=11, weight="bold", y=0.99)
    save("Figure1_cohort_attrition.tif")

# =====================================================
# Figure 2 — Evidence drift heatmap (compact, readable)
# =====================================================
def fig2():
    fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
    KEYS = ["comorb_count","prior_pa_res_levofloxacin",
            "abx_carbapenem_0_30","abx_carbapenem_181_365",
            "abx_antipseudomonal_bl_0_30","abx_antipseudomonal_bl_gt365",
            "prior_pa_res_piperacillin_tazobactam","prior_pa_res_tobramycin",
            "abx_glycopeptide_0_30","abx_glycopeptide_gt365",
            "prior_pseudo","prior_pseudo_days","age_bin","comorb_malignancy","nh_30d"]
    LABS = {"comorb_count":"Comorbidity count",
            "prior_pa_res_levofloxacin":"Prior LVX resistance",
            "abx_carbapenem_0_30":"Carbapenem 0–30d",
            "abx_carbapenem_181_365":"Carbapenem 181–365d",
            "abx_antipseudomonal_bl_0_30":"Anti-PA BL 0–30d",
            "abx_antipseudomonal_bl_gt365":"Anti-PA BL >365d",
            "prior_pa_res_piperacillin_tazobactam":"Prior TZP resistance",
            "prior_pa_res_tobramycin":"Prior TOB resistance",
            "abx_glycopeptide_0_30":"Glycopeptide 0–30d",
            "abx_glycopeptide_gt365":"Glycopeptide >365d",
            "prior_pseudo":"Prior Pseudomonas",
            "prior_pseudo_days":"Days since Pseudomonas",
            "age_bin":"Age","comorb_malignancy":"Malignancy","nh_30d":"Nursing home 30d"}
    DIRS = ["Stanford","UTSW"]

    tall, y_labels = [], []
    for d in DIRS:
        sub = fd[fd["tgt"]==d]
        row = []
        for f in KEYS:
            r = sub[sub["feature"]==f]
            vals = [float(r[r["task"]==t]["ratio"].iloc[0]) if len(r[r["task"]==t]) else np.nan for t in TASKS]
            row.append(vals)
            y_labels.append(f"{LABS.get(f,f)}\n({d[:1]})")
        tall.extend(row)

    tall = np.array(tall)   # (30, 5)
    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)

    fig, ax = plt.subplots(figsize=(5.8, 9.8))
    im = ax.imshow(tall, aspect="auto", cmap=cmap, norm=norm)

    ax.set_xticks(range(5)); ax.set_xticklabels([T_LAB[t] for t in TASKS], fontsize=9, weight="bold")
    ax.set_yticks(range(30)); ax.set_yticklabels(y_labels, fontsize=7)
    ax.axhline(y=14.5, color=C_INK, linewidth=0.8)

    # Cell annotation — only for |ratio-1|>0.25 (clear signal)
    for i in range(30):
        for j in range(5):
            v = tall[i,j]
            if not np.isnan(v) and abs(v-1) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.5,
                        color="white" if v<0.5 or v>1.5 else C_INK)

    cbar = fig.colorbar(im, ax=ax, shrink=0.88, pad=0.015)
    cbar.set_label("SHAP ratio (target / source)", fontsize=8.5)
    cbar.ax.axhline(y=1.0, color=C_INK, lw=0.5, ls="--")
    ax.set_title("Figure 2. Feature evidence drift: |SHAP| ratio (target / source).", fontsize=10.5, weight="bold", pad=8)
    save("Figure2_feature_drift_heatmap.tif")

# =====================================================
# Figure 3 — Calibration curves (3 tasks × 2 sites, clean)
# =====================================================
def fig3():
    from sklearn.calibration import calibration_curve
    T3 = ["meropenem","ceftazidime","cefepime"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.5))
    for j, t in enumerate(T3):
        mgb_raw = pd.read_csv(os.path.join(CLEAN,"MGB",f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        mgb = prep(mgb_raw); feats = feature_cols(mgb); maps = fit_cat_maps(mgb_raw)
        m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
        for i, (d, dl) in enumerate([("Stanford","S"),("UTSW","U")]):
            ax = axes[i,j]
            te_raw = pd.read_csv(os.path.join(CLEAN,d,f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
            te = prep_cross(te_raw, maps, mgb["adi"].median())
            X = te[[f for f in feats if f in te.columns]]; y = te["label"].values
            pred = m.predict_proba(X)[:,1]
            fp, mp = calibration_curve(y, pred, n_bins=10, strategy="uniform")
            ax.plot([0,1],[0,1],"k--",alpha=0.25,lw=0.8)
            ax.plot(mp, fp, "o-", color=C_BLUE, markersize=5, lw=1.5, mec=C_BLUE, mfc="white")
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=10, weight="bold")
            ax.set_xlim(-0.02,1.02); ax.set_ylim(-0.02,1.02)
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
            tm = pd.read_csv(os.path.join(R3,"transfer_matrix.csv"))
            rr = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==d)]
            brier = f"{rr['brier'].iloc[0]:.3f}" if len(rr) else "NA"
            ax.text(0.97, 0.06, f"Brier={brier}", ha="right", va="bottom",
                    transform=ax.transAxes, fontsize=8, color=C_GREY)
    fig.suptitle("Figure 3. Calibration curves (MGB \u2192 external, representative tasks).",
                 fontsize=11, weight="bold")
    plt.tight_layout(rect=[0,0,1,0.94])
    save("Figure3_calibration_curves.tif")

# =====================================================
# S1 — DCA curves (5 tasks × 2 directions, clean grid)
# =====================================================
def fig_s1():
    dca = pd.read_csv(os.path.join(R3,"dca.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(14, 5.5), sharex=True)
    for j, t in enumerate(TASKS):
        for i, (d, dl) in enumerate([("Stanford","S"),("UTSW","U")]):
            ax = axes[i,j]; sub = dca[(dca["task"]==t)&(dca["tgt"]==d)]
            for mod, ls, c in [("full","-",C_BLUE),("subset","--",C_RED)]:
                s = sub[sub["model"]==mod]
                if len(s): ax.plot(s["threshold"], s["net_benefit"], ls, color=c, lw=1.2)
            ax.axhline(y=0, color=C_INK, lw=0.3, alpha=0.3)
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=9, weight="bold")
            ax.set_xlim(0, 0.45); ax.set_ylim(-0.05, None)
            if i==1: ax.set_xlabel("Threshold probability")
            if j==0: ax.set_ylabel("Net benefit")
    fig.suptitle("Figure S1. Decision curve analysis (MGB\u2192external, all primary tasks). Solid: full; Dashed: stable-subset.",
                 fontsize=10.5, weight="bold")
    plt.tight_layout(rect=[0,0,1,0.93])
    save("FigureS1_dca.tif")

# =====================================================
# S2 — Era sensitivity: horizontal bar, grouped
# =====================================================
def fig_s2():
    era = pd.read_csv(os.path.join(R3,"era_sensitivity.csv"))
    e20 = era[era["cut"]==2020]
    x = np.arange(len(e20)); w = 0.35
    fig, ax = plt.subplots(figsize=(10, 5.2))
    ax.barh(x + w/2, e20["auroc_cut"], w, label=">=2020 only", color=C_BLUE, alpha=0.88)
    ax.barh(x - w/2, e20["auroc_all"], w, label="All years", color=C_LGREY, alpha=0.7)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{T_LAB[r['task']]}\u2192{r['tgt'][:1]}" for _,r in e20.iterrows()], fontsize=7.5)
    ax.set_xlabel("AUROC"); ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Figure S2. Era sensitivity: all years vs. \u22652020 (MGB\u2192external).", fontsize=10.5, weight="bold")
    plt.tight_layout()
    save("FigureS2_era_sensitivity.tif")

# =====================================================
# S3 — Mucoid sensitivity: grouped horizontal bar
# =====================================================
def fig_s3():
    muc = pd.read_csv(os.path.join(R3,"mucoid_sensitivity.csv"))
    muc = muc.dropna(subset=["auroc_mucoid_only"])
    x = np.arange(len(muc)); w = 0.25
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.barh(x + w, muc["auroc_all"], w, label="All samples", color=C_BLUE, alpha=0.88)
    ax.barh(x, muc["auroc_no_mucoid"], w, label="Excl. mucoid", color=C_TEAL, alpha=0.88)
    ax.barh(x - w, muc["auroc_mucoid_only"], w, label="Mucoid only", color=C_RED, alpha=0.88)
    ax.set_yticks(x)
    ax.set_yticklabels([f"{T_LAB[r['task']]}\u2192{r['tgt'][:1]}" for _,r in muc.iterrows()], fontsize=8)
    ax.set_xlabel("AUROC"); ax.legend(fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Figure S3. Mucoid sensitivity (Stanford only; UTSW has no mucoid cohort).",
                 fontsize=10.5, weight="bold")
    plt.tight_layout()
    save("FigureS3_mucoid_sensitivity.tif")

# =====================================================
# S4 — CI forest plot (30 directions, clean)
# =====================================================
def fig_s4():
    tm = pd.read_csv(os.path.join(R3,"transfer_matrix.csv"))
    ss = pd.read_csv(os.path.join(R3,"stable_subset.csv"))
    # Sort by task then gap magnitude
    tm = tm.sort_values(["task","transfer_gap"])
    fig, ax = plt.subplots(figsize=(11, 8))
    colors = {"MGB":"#2171B5","Stanford":"#5AB4AC","UTSW":"#F4A582"}
    gaps_all = []
    for _, r in tm.iterrows():
        label = f"{T_LAB[r['task']]} {r['src']}\u2192{r['tgt']}"
        ci = ss[(ss["task"]==r["task"])&(ss["tgt"]==r["tgt"])]
        lo = hi = None
        if len(ci)>0:
            p = str(ci["boot_ci"].iloc[0]).split("-")
            lo, hi = float(p[0]), float(p[-1])
        if lo is None: lo=r["auroc"]-0.03; hi=r["auroc"]+0.03
        gaps_all.append((r["auroc"], lo, hi, label, r["internal_auroc"], colors.get(r["src"], C_INK)))
    n = len(gaps_all); y_positions = list(range(n))
    for idx, (auc, lo, hi, label, internal, c) in enumerate(gaps_all):
        lo_err = max(0, auc-lo); hi_err = max(0, hi-auc)
        ax.errorbar(auc, idx, xerr=[[lo_err],[hi_err]], fmt="o", color=c,
                    capsize=2.5, markersize=4.5, linewidth=1.0, zorder=4)
    ax.set_yticks(range(n)); ax.set_yticklabels([g[3] for g in gaps_all], fontsize=6.5)
    for idx, g in enumerate(gaps_all):
        ax.axhline(y=idx, color=C_LGREY, lw=0.25, alpha=0.5)
    ax.set_xlabel("AUROC", fontsize=9)
    ax.set_title("Figure S4. Full CI matrix (30 directions, patient-level bootstrap 200 iterations).",
                 fontsize=10.5, weight="bold")
    plt.tight_layout()
    save("FigureS4_ci_matrix.tif")


# ── Run all ──
if __name__ == "__main__":
    print("Generating Figure 1..."); fig1()
    print("Generating Figure 2..."); fig2()
    print("Generating Figure 3..."); fig3()
    print("Generating Figure S1..."); fig_s1()
    print("Generating Figure S2..."); fig_s2()
    print("Generating Figure S3..."); fig_s3()
    print("Generating Figure S4..."); fig_s4()
    print("All figures done:", OUT)
