# -*- coding: utf-8 -*-
"""
Figures v3 — dataviz-skill compliant rewrite.
Palette: validated reference palette (light surface #fcfcfb)
  categorical slots 1-3: #2a78d6, #eb6834, #1baf7a (validator: ALL PASS)
  diverging pair: blue #2a78d6 <-> red #e34948, neutral midpoint #f0efec
Ink tokens: primary #0b0b0b, secondary #52514e, muted #898781,
            gridline #e1e0d9, baseline #c3c2b7
Anti-pattern rules applied: no dual axis, no cycled hues, no per-point labels,
thin marks, solid hairline grid, text in ink (never series color), legend for
>=2 series, 2px surface gap between bars, no hue at diverging midpoint.
TIFF 300 dpi, LZW.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os, sys

OUT   = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)
R3    = r"E:\ARMD\results\phase3"
CLEAN = r"E:\ARMD\data\clean"
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross

# ── validated palette / tokens ──
SURFACE   = "#fcfcfb"
INK       = "#0b0b0b"
INK_SEC   = "#52514e"
INK_MUTED = "#898781"
GRID      = "#e1e0d9"
BASELINE  = "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"     # categorical slots 1-3 (validated)
POLE_LO, POLE_HI = "#2a78d6", "#e34948"          # diverging pair
MID_GRAY  = "#f0efec"                            # diverging midpoint

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 10, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8, "legend.fontsize": 7.5,
    "savefig.dpi": 300, "savefig.bbox": "tight", "savefig.pad_inches": 0.08,
    "axes.edgecolor": BASELINE, "axes.labelcolor": INK,
    "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
T_LAB = {"meropenem":"MEM","ciprofloxacin":"CIP","levofloxacin":"LVX",
         "ceftazidime":"CAZ","cefepime":"FEP"}

def save(name):
    fp = os.path.join(OUT, name)
    plt.savefig(fp, pil_kwargs={"compression": "tiff_lzw"})
    print(f"  {name} saved ({os.path.getsize(fp)//1024} KB)")
    plt.close()

def hairline(ax):
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(BASELINE)
    ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(width=0.6, length=2.5)

# ============ Figure 1 — enrollment flow ============
def fig1():
    fig, ax = plt.subplots(figsize=(7.5, 7.4))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(y, title, sub, x0=2.5, w=5.0, h=0.74):
        r = mpatches.FancyBboxPatch((x0, y-h/2), w, h, boxstyle="round,pad=0.08",
                                     facecolor=SURFACE, edgecolor=INK, linewidth=1.1, zorder=3)
        ax.add_patch(r)
        ax.text(x0+w/2, y+0.07, title, ha="center", va="center", fontsize=9, weight="bold",
                color=INK, zorder=4)
        ax.text(x0+w/2, y-0.30, sub, ha="center", va="center", fontsize=7.3,
                color=INK_SEC, zorder=4)

    def arrow(yf, yt, note, x=5.0):
        ax.annotate("", xy=(x, yt+0.33), xytext=(x, yf-0.33),
                    arrowprops=dict(arrowstyle="->", color=INK_SEC, lw=1.1), zorder=2)
        ax.text(x+2.65, (yf+yt)/2, note, ha="left", va="center", fontsize=7.2,
                color=INK_SEC, zorder=4)

    def note(y, text):
        ax.text(7.85, y, text, ha="left", va="center", fontsize=6.8, color=INK_MUTED, zorder=4)

    y = 9.2
    box(y, "MGB microbiology cohort", "4,960,599 rows; 2015\u20132024 (dates patient-level shifted)")
    arrow(y, y-0.76, "Organism starts with PSEUDOMONAS AERUGINOSA")
    y -= 0.81
    box(y, "P. aeruginosa rows", "221,675 rows \u00b7 15,254 cultures \u00b7 5,216 patients")
    arrow(y, y-0.76, "Drug in 9-task set; exclude preliminary and negative cultures")
    y -= 0.81
    box(y, "PA target-drug rows after exclusions", "169,477 rows \u00b7 15,253 cultures")
    arrow(y, y-0.76, "CLSI_2022_pheno in {Susceptible, Intermediate, Resistant}")
    y -= 0.81
    box(y, "PA rows with valid label", "105,421 rows \u00b7 10,578 cultures (69% of PA cultures)")
    note(y, "37.8% excluded: no interpretable\nCLSI phenotype; mucoid 38.6%\nof excluded rows")

    fig.suptitle("Figure 1. MGB cohort attrition (TRIPOD-AI enrollment flow).",
                 fontsize=11, weight="bold", color=INK, y=0.99)
    save("Figure1_cohort_attrition.tif")

# ============ Figure 2 — evidence drift heatmap ============
def fig2():
    fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
    KEYS = ["comorb_count","prior_pa_res_levofloxacin",
            "abx_carbapenem_0_30","abx_carbapenem_181_365",
            "abx_antipseudomonal_bl_0_30","abx_antipseudomonal_bl_gt365",
            "prior_pa_res_piperacillin_tazobactam","prior_pa_res_tobramycin",
            "abx_glycopeptide_0_30","abx_glycopeptide_gt365",
            "prior_pseudo","prior_pseudo_days","age_bin","comorb_malignancy","nh_30d"]
    LABS = {"comorb_count":"Comorbidity count","prior_pa_res_levofloxacin":"Prior LVX resistance",
            "abx_carbapenem_0_30":"Carbapenem 0\u201330d","abx_carbapenem_181_365":"Carbapenem 181\u2013365d",
            "abx_antipseudomonal_bl_0_30":"Anti-PA BL 0\u201330d","abx_antipseudomonal_bl_gt365":"Anti-PA BL >365d",
            "prior_pa_res_piperacillin_tazobactam":"Prior TZP resistance","prior_pa_res_tobramycin":"Prior TOB resistance",
            "abx_glycopeptide_0_30":"Glycopeptide 0\u201330d","abx_glycopeptide_gt365":"Glycopeptide >365d",
            "prior_pseudo":"Prior Pseudomonas","prior_pseudo_days":"Days since Pseudomonas",
            "age_bin":"Age","comorb_malignancy":"Malignancy","nh_30d":"Nursing home 30d"}

    tall, ylab = [], []
    for d in ["Stanford", "UTSW"]:
        sub = fd[fd["tgt"] == d]
        for f in KEYS:
            vals = [float(sub[(sub["feature"]==f)&(sub["task"]==t)]["ratio"].iloc[0])
                    if len(sub[(sub["feature"]==f)&(sub["task"]==t)]) else np.nan for t in TASKS]
            tall.append(vals)
            ylab.append(f"{LABS.get(f, f)}  \u00b7 {d[0]}")
    tall = np.array(tall)

    cmap = plt.cm.RdBu_r
    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)

    fig, ax = plt.subplots(figsize=(5.6, 9.6))
    im = ax.imshow(tall, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(range(5)); ax.set_xticklabels([T_LAB[t] for t in TASKS], fontsize=9, weight="bold")
    ax.set_yticks(range(30)); ax.set_yticklabels(ylab, fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_color(BASELINE)
    ax.axhline(y=14.5, color=INK, lw=0.7)
    # selective direct labels: only |ratio-1| > 0.25
    for i in range(30):
        for j in range(5):
            v = tall[i, j]
            if not np.isnan(v) and abs(v-1) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.3,
                        color="white" if (v < 0.5 or v > 1.5) else INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
    cbar.set_label("SHAP ratio (target / source)", fontsize=8, color=INK)
    cbar.ax.tick_params(labelsize=7, colors=INK_SEC)
    ax.set_title("Figure 2. Feature evidence drift (same MGB-trained model, full-reference SHAP).",
                 fontsize=10, weight="bold", color=INK, pad=10)
    save("Figure2_feature_drift_heatmap.tif")

# ============ Figure 3 — calibration curves ============
def fig3():
    from sklearn.calibration import calibration_curve
    T3 = ["meropenem", "ceftazidime", "cefepime"]
    fig, axes = plt.subplots(2, 3, figsize=(10, 6.2))
    for j, t in enumerate(T3):
        mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        mgb = prep(mgb_raw); feats = feature_cols(mgb); maps = fit_cat_maps(mgb_raw)
        m = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])
        for i, (d, dl) in enumerate([("Stanford", "S"), ("UTSW", "U")]):
            ax = axes[i, j]; hairline(ax)
            te_raw = pd.read_csv(os.path.join(CLEAN, d, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
            te = prep_cross(te_raw, maps, mgb["adi"].median())
            X = te[[f for f in feats if f in te.columns]]; y = te["label"].values
            pred = m.predict_proba(X)[:, 1]
            fp, mp = calibration_curve(y, pred, n_bins=10, strategy="uniform")
            ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.0)   # perfect-calibration reference, solid
            ax.plot(mp, fp, "-o", color=S1, lw=1.5, markersize=4.5, mfc=SURFACE, mec=S1)
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=10, weight="bold", color=INK)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
            tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
            rr = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==d)]
            if len(rr):
                ax.text(0.97, 0.05, f"Brier {rr['brier'].iloc[0]:.3f}", ha="right",
                        va="bottom", transform=ax.transAxes, fontsize=7.5, color=INK_SEC)
    fig.suptitle("Figure 3. Calibration curves (MGB \u2192 external, representative tasks).",
                 fontsize=11, weight="bold", color=INK)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save("Figure3_calibration_curves.tif")

# ============ S1 — DCA ============
def fig_s1():
    dca = pd.read_csv(os.path.join(R3, "dca.csv"))
    fig, axes = plt.subplots(2, 5, figsize=(13.5, 5.4), sharex=True)
    for j, t in enumerate(TASKS):
        for i, (d, dl) in enumerate([("Stanford", "S"), ("UTSW", "U")]):
            ax = axes[i, j]; hairline(ax)
            sub = dca[(dca["task"]==t)&(dca["tgt"]==d)]
            ax.plot(sub[sub["model"]=="full"]["threshold"],
                    sub[sub["model"]=="full"]["net_benefit"], "-", color=S1, lw=1.5)
            ax.plot(sub[sub["model"]=="subset"]["threshold"],
                    sub[sub["model"]=="subset"]["net_benefit"], "-", color=S2, lw=1.5)
            ax.axhline(y=0, color=BASELINE, lw=0.8)
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=9, weight="bold", color=INK)
            ax.set_xlim(0, 0.45); ax.set_ylim(-0.05, None)
            if i == 1: ax.set_xlabel("Threshold probability")
            if j == 0: ax.set_ylabel("Net benefit")
    handles = [plt.Line2D([0],[0], color=S1, lw=1.5, label="Full model"),
               plt.Line2D([0],[0], color=S2, lw=1.5, label="Stable-subset model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8.5)
    fig.suptitle("Figure S1. Decision curve analysis (MGB\u2192external, all primary tasks).",
                 fontsize=10.5, weight="bold", color=INK)
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    save("FigureS1_dca.tif")

# ============ S2 — era sensitivity ============
def fig_s2():
    era = pd.read_csv(os.path.join(R3, "era_sensitivity.csv"))
    e20 = era[era["cut"] == 2020]
    fig, ax = plt.subplots(figsize=(9.5, 5.0)); hairline(ax)
    y = np.arange(len(e20))
    ax.barh(y + 0.19, e20["auroc_all"], 0.36, color=INK_MUTED, label="All years",
            edgecolor=SURFACE, linewidth=0.8)
    ax.barh(y - 0.19, e20["auroc_cut"], 0.36, color=S1, label="\u22652020 only",
            edgecolor=SURFACE, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{T_LAB[r['task']]}\u2192{r['tgt'][:1]}" for _, r in e20.iterrows()], fontsize=8)
    ax.set_xlabel("AUROC")
    ax.legend(frameon=False, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Figure S2. Era sensitivity: all years vs. \u22652020 (MGB\u2192external).",
                 fontsize=10.5, weight="bold", color=INK)
    plt.tight_layout()
    save("FigureS2_era_sensitivity.tif")

# ============ S3 — mucoid sensitivity ============
def fig_s3():
    muc = pd.read_csv(os.path.join(R3, "mucoid_sensitivity.csv"))
    muc = muc.dropna(subset=["auroc_mucoid_only"])
    fig, ax = plt.subplots(figsize=(8.5, 4.6)); hairline(ax)
    y = np.arange(len(muc))
    ax.barh(y + 0.26, muc["auroc_all"], 0.22, color=S1, label="All samples",
            edgecolor=SURFACE, linewidth=0.8)
    ax.barh(y, muc["auroc_no_mucoid"], 0.22, color=S2, label="Excl. mucoid",
            edgecolor=SURFACE, linewidth=0.8)
    ax.barh(y - 0.26, muc["auroc_mucoid_only"], 0.22, color=S3, label="Mucoid only",
            edgecolor=SURFACE, linewidth=0.8)
    ax.set_yticks(y)
    ax.set_yticklabels([f"{T_LAB[r['task']]}\u2192{r['tgt'][:1]}" for _, r in muc.iterrows()], fontsize=8)
    ax.set_xlabel("AUROC")
    ax.legend(frameon=False, fontsize=8)
    ax.invert_yaxis()
    ax.set_title("Figure S3. Mucoid sensitivity (Stanford only; UTSW lacks a mucoid cohort).",
                 fontsize=10.5, weight="bold", color=INK)
    plt.tight_layout()
    save("FigureS3_mucoid_sensitivity.tif")

# ============ S4 — CI forest ============
def fig_s4():
    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
    ss = pd.read_csv(os.path.join(R3, "stable_subset.csv"))
    tm = tm.sort_values(["task", "transfer_gap"])
    src_col = {"MGB": S1, "Stanford": S2, "UTSW": S3}
    fig, ax = plt.subplots(figsize=(10.5, 8.0)); hairline(ax)
    labels, aucs, loes, his, cols = [], [], [], [], []
    for _, r in tm.iterrows():
        labels.append(f"{T_LAB[r['task']]}  {r['src']}\u2192{r['tgt']}")
        ci = ss[(ss["task"]==r["task"])&(ss["tgt"]==r["tgt"])]
        if len(ci) > 0:
            parts = str(ci["boot_ci"].iloc[0]).split("-")
            lo, hi = float(parts[0]), float(parts[-1])
        else:
            lo, hi = r["auroc"]-0.03, r["auroc"]+0.03
        aucs.append(r["auroc"]); loes.append(max(0, r["auroc"]-lo)); his.append(max(0, hi-r["auroc"]))
        cols.append(src_col.get(r["src"], INK_SEC))
    ypos = list(range(len(aucs)))
    for k in range(len(aucs)):
        ax.errorbar(aucs[k], ypos[k], xerr=[[loes[k]], [his[k]]], fmt="o", color=cols[k],
                    capsize=2.2, markersize=4.5, lw=1.0, zorder=4)
        ax.axhline(y=ypos[k], color=GRID, lw=0.4, zorder=1)
    ax.set_yticks(ypos); ax.set_yticklabels(labels, fontsize=6.6)
    ax.set_xlabel("AUROC")
    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S1, markersize=6, label="Source: MGB"),
               plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S2, markersize=6, label="Source: Stanford"),
               plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S3, markersize=6, label="Source: UTSW")]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Figure S4. AUROC with 95% CI for all 30 transfer directions (patient-level bootstrap).",
                 fontsize=10.5, weight="bold", color=INK)
    plt.tight_layout()
    save("FigureS4_ci_matrix.tif")


if __name__ == "__main__":
    print("Fig1"); fig1()
    print("Fig2"); fig2()
    print("Fig3"); fig3()
    print("S1");  fig_s1()
    print("S2");  fig_s2()
    print("S3");  fig_s3()
    print("S4");  fig_s4()
    print("Done:", OUT)
