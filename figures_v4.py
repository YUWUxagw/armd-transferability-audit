# -*- coding: utf-8 -*-
"""
Figures v4 — nature-figure + dataviz dual-skill compliant.

Every figure carries its FIGURE CONTRACT (core conclusion / evidence chain /
archetype / export contract) before code, per nature-figure/static/core/contract.md.
Palette: validated reference palette (dataviz validator, all gates PASS):
  categorical slots 1-3  #2a78d6 #eb6834 #1baf7a
  diverging blue #2a78d6 <-> red #e34948, midpoint #f0efec
Ink: primary #0b0b0b / secondary #52514e / muted #898781 / gridline #e1e0d9 / baseline #c3c2b7
Export: SVG (editable text) + PDF (fonttype 42) + TIFF 300dpi LZW — per backend fragment.

Anti-pattern rules applied (dataviz): no dual axis; no cycled hues; no per-point
labels; thin marks; solid hairline grid; text in ink only; legend for >=2 series;
2px surface gap between bars; no hue at diverging midpoint.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import pandas as pd
import os, sys

def _pick(*cands):
    r"""2026-09-20：原始数据（data/clean）在 E:\ARMD，结果 CSV 两边都有；按存在性自动择一。"""
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[-1]

OUT   = r"F:\E\Machine Learning\ARMD\02_图表"            # 交付位置（图件随项目走）
os.makedirs(OUT, exist_ok=True)
R3    = _pick(r"E:\ARMD\results\phase3", r"F:\E\Machine Learning\ARMD\05_源数据\phase3")
CLEAN = _pick(r"E:\ARMD\data\clean", r"F:\E\Machine Learning\ARMD\05_源数据\clean")
sys.path.insert(0, os.path.dirname(__file__))
try:   # 2026-09-20：数据依赖模块（需 sklearn/原始数据）；示意图（Fig1）无需，缺失时降级为警告
    from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross
except ImportError as _e:
    print(f"[warn] step14_model_phase2 未导入（{_e}）；仅数据依赖型图需要它")

SURFACE, INK, INK_SEC, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POLE_LO, POLE_HI, MID_GRAY = "#2a78d6", "#e34948", "#f0efec"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "svg.fonttype": "none", "pdf.fonttype": 42,
    "font.size": 7, "axes.titlesize": 9.5, "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5, "ytick.labelsize": 7.5, "legend.fontsize": 7,
    "axes.spines.right": False, "axes.spines.top": False, "axes.linewidth": 0.8,
    "legend.frameon": False, "savefig.dpi": 600, "savefig.bbox": "tight",
    "savefig.pad_inches": 0.08, "axes.edgecolor": BASELINE,
    "axes.labelcolor": INK, "xtick.color": INK_SEC, "ytick.color": INK_SEC,
    "text.color": INK, "figure.facecolor": SURFACE, "axes.facecolor": SURFACE,
})

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
T_LAB = {"meropenem":"MEM","ciprofloxacin":"CIP","levofloxacin":"LVX",
         "ceftazidime":"CAZ","cefepime":"FEP"}

def save(name):
    base = os.path.join(OUT, name)
    fig = plt.gcf()
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".tif", pil_kwargs={"compression": "tiff_lzw"})
    print(f"  {name} (svg/pdf/tif; tif {os.path.getsize(base + '.tif')//1024} KB)")
    plt.close()

def hairline(ax):
    ax.spines["left"].set_color(BASELINE); ax.spines["bottom"].set_color(BASELINE)
    ax.tick_params(width=0.6, length=2.5)

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 1 — enrollment flow
# CONTRACT: core conclusion: the PA cohort shrinks from 4.96M rows to 10,578
#   cultures (69% of PA cultures); the dominant loss is label validity, and
#   mucoid isolates are disproportionately excluded. Role: cohort construction
#   transparency (TRIPOD-AI). Archetype: schematic-led composite.
# Evidence chain: 4 attrition steps + side note on mucoid composition.
# ═══════════════════════════════════════════════════════════════════════════
def fig1():
    fig, ax = plt.subplots(figsize=(7.4, 7.3))
    ax.set_xlim(0, 10); ax.set_ylim(0, 10); ax.axis("off")

    def box(y, title, sub, x0=2.4, w=5.0, h=0.74):
        r = mpatches.FancyBboxPatch((x0, y-h/2), w, h, boxstyle="round,pad=0.08",
                                     facecolor=SURFACE, edgecolor=INK, linewidth=1.0, zorder=3)
        ax.add_patch(r)
        ax.text(x0+w/2, y+0.07, title, ha="center", va="center", fontsize=8.5,
                weight="bold", color=INK, zorder=4)
        ax.text(x0+w/2, y-0.30, sub, ha="center", va="center", fontsize=7.0,
                color=INK_SEC, zorder=4)

    def arrow(yf, yt, note, x=5.0):
        ax.annotate("", xy=(x, yt+0.33), xytext=(x, yf-0.33),
                    arrowprops=dict(arrowstyle="->", color=INK_SEC, lw=1.0), zorder=2)
        ax.text(x+2.7, (yf+yt)/2, note, ha="left", va="center", fontsize=6.9,
                color=INK_SEC, zorder=4)

    def note(y, text):
        ax.text(7.9, y, text, ha="left", va="center", fontsize=6.5, color=INK_MUTED, zorder=4)

    y = 9.2
    box(y, "MGB microbiology cohort", "4,960,599 rows; 2015\u20132024 (dates shifted)")
    arrow(y, y-0.76, "Organism: PSEUDOMONAS AERUGINOSA")
    y -= 0.81
    box(y, "P. aeruginosa rows", "221,675 rows \u00b7 15,254 cultures \u00b7 5,216 patients")
    arrow(y, y-0.76, "9-task drugs; exclude preliminary / negative cultures")
    y -= 0.81
    box(y, "PA target-drug rows after exclusions", "169,477 rows \u00b7 15,253 cultures")
    arrow(y, y-0.76, "CLSI_2022_pheno in {S, I, R}")
    y -= 0.81
    box(y, "PA rows with valid label", "105,421 rows \u00b7 10,578 cultures (69%)")
    note(y, "37.8% excluded: no interpretable\nCLSI phenotype; mucoid = 38.6%\nof label-validity-excluded rows")

    fig.suptitle("Figure 1. MGB cohort attrition (TRIPOD-AI enrollment flow).",
                 fontsize=10.5, weight="bold", color=INK, y=0.99)
    save("Figure1_cohort_attrition")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 2 — feature evidence drift heatmap
# CONTRACT: core conclusion: within-site-stable features split into preserved
#   (comorbidity count, prior LVX resistance) and attenuated classes (exposure
#   windows, prior TOB/TZP resistance) upon transfer; ratio center 1.0.
#   Role: validation of the evidence-drift claim. Archetype: quantitative grid.
# Evidence chain: 15 features x 5 tasks x 2 target sites; cell labels only for
#   |ratio-1|>0.25 (selective direct labels).
# ═══════════════════════════════════════════════════════════════════════════
def fig2():
    fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
    KEYS = ["comorb_count","prior_pa_res_levofloxacin",
            "abx_carbapenem_0_30","abx_carbapenem_181_365",
            "abx_antipseudomonal_bl_0_30","abx_antipseudomonal_bl_gt365",
            "prior_pa_res_piperacillin_tazobactam","prior_pa_res_tobramycin",
            "prior_pa_res_amikacin",   # 2026-09-20 补入：现版 Fig2 含此特征（16 个），
                                       # 缺它则重跑会把图降级为 15 特征、与图注/§4.9 相悖
            "abx_glycopeptide_0_30","abx_glycopeptide_gt365",
            "prior_pseudo","prior_pseudo_days","age_bin","comorb_malignancy","nh_30d"]
    LABS = {"comorb_count":"Comorbidity count","prior_pa_res_levofloxacin":"Prior LVX resistance",
            "abx_carbapenem_0_30":"Carbapenem 0\u201330d","abx_carbapenem_181_365":"Carbapenem 181\u2013365d",
            "abx_antipseudomonal_bl_0_30":"Anti-PA BL 0\u201330d","abx_antipseudomonal_bl_gt365":"Anti-PA BL >365d",
            "prior_pa_res_piperacillin_tazobactam":"Prior TZP resistance","prior_pa_res_tobramycin":"Prior TOB resistance",
            "prior_pa_res_amikacin":"Prior AMK resistance",
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

    norm = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.0)
    fig, ax = plt.subplots(figsize=(5.6, 9.6))
    im = ax.imshow(tall, aspect="auto", cmap=plt.cm.RdBu_r, norm=norm)
    ax.set_xticks(range(5)); ax.set_xticklabels([T_LAB[t] for t in TASKS], fontsize=8.5, weight="bold")
    n_row = len(ylab)   # 2026-09-20：由硬编码 30/14.5 改为长度自适应（补入 AMK 后为 32）
    ax.set_yticks(range(n_row)); ax.set_yticklabels(ylab, fontsize=6.6)
    ax.tick_params(length=0)
    for sp in ax.spines.values(): sp.set_color(BASELINE)
    ax.axhline(y=len(KEYS) - 0.5, color=INK, lw=0.7)
    for i in range(n_row):
        for j in range(5):
            v = tall[i, j]
            if not np.isnan(v) and abs(v-1) > 0.25:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=6.2,
                        color="white" if (v < 0.5 or v > 1.5) else INK)
    cbar = fig.colorbar(im, ax=ax, shrink=0.9, pad=0.015)
    cbar.set_label("SHAP ratio (target / source)", fontsize=7.5, color=INK)
    cbar.ax.tick_params(labelsize=6.5, colors=INK_SEC)
    ax.set_title("Figure 3. Feature evidence drift (same MGB-trained model, full-reference SHAP).",
                 fontsize=9.5, weight="bold", color=INK, pad=10)
    save("Figure3_feature_drift_heatmap")

# ═══════════════════════════════════════════════════════════════════════════
# FIGURE 3 — calibration curves
# CONTRACT: core conclusion: cross-system calibration collapses (slopes <1,
#   negative intercepts), with FEP→U worst (Brier 0.270). Role: validation of
#   the calibration-collapse claim. Archetype: quantitative grid.
# Evidence chain: 3 representative tasks x 2 directions; each panel shows the
#   calibration curve vs. perfect-calibration reference, Brier annotated.
# ═══════════════════════════════════════════════════════════════════════════
def fig3():
    from sklearn.calibration import calibration_curve
    T3 = ["meropenem", "ceftazidime", "cefepime"]
    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
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
            ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.0)
            ax.plot(mp, fp, "-o", color=S1, lw=1.5, markersize=4, mfc=SURFACE, mec=S1)
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=9, weight="bold", color=INK)
            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)
            ax.set_xlabel("Predicted probability"); ax.set_ylabel("Observed fraction")
            rr = tm[(tm["task"]==t)&(tm["src"]=="MGB")&(tm["tgt"]==d)]
            if len(rr):
                ax.text(0.97, 0.05, f"Brier {rr['brier'].iloc[0]:.3f}", ha="right", va="bottom",
                        transform=ax.transAxes, fontsize=7, color=INK_SEC)
    fig.suptitle("Figure 2. Calibration curves (MGB \u2192 external, the three largest-gap tasks).",
                 fontsize=10.5, weight="bold", color=INK)
    plt.tight_layout(rect=[0, 0, 1, 0.94])
    save("Figure2_calibration_curves")

# ═══════════════════════════════════════════════════════════════════════════
# S1 — DCA
# CONTRACT: core conclusion: cross-system net benefit narrows sharply at
#   clinically common thresholds (t=0.20); stable-subset models do not
#   systematically rescue it. Role: robustness. Archetype: quantitative grid.
# Evidence chain: 5 tasks x 2 directions; full vs subset lines (slots 1-2),
#   single shared legend.
# ═══════════════════════════════════════════════════════════════════════════
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
            ax.set_title(f"{T_LAB[t]} \u2192 {dl}", fontsize=8.5, weight="bold", color=INK)
            ax.set_xlim(0, 0.45); ax.set_ylim(-0.05, None)
            if i == 1: ax.set_xlabel("Threshold probability")
            if j == 0: ax.set_ylabel("Net benefit")
    handles = [plt.Line2D([0],[0], color=S1, lw=1.5, label="Full model"),
               plt.Line2D([0],[0], color=S2, lw=1.5, label="Stable-subset model")]
    fig.legend(handles=handles, loc="lower center", ncol=2, frameon=False, fontsize=8)
    fig.suptitle("Figure S3. Decision curve analysis (MGB\u2192external, all primary tasks).",
                 fontsize=10, weight="bold", color=INK)
    plt.tight_layout(rect=[0, 0.07, 1, 0.93])
    save("FigureS3_dca")

# ═══════════════════════════════════════════════════════════════════════════
# S2 — era sensitivity
# CONTRACT: core conclusion: restricting to the breakpoint-stable era (>=2020)
#   modestly improves most directions but gaps persist. Role: robustness.
# Archetype: quantitative grid (grouped horizontal bars; 2px surface gaps).
# ═══════════════════════════════════════════════════════════════════════════
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
    ax.set_yticklabels([f"{T_LAB[r['task']]}\u2192{r['tgt'][:1]}" for _, r in e20.iterrows()], fontsize=7.5)
    ax.set_xlabel("AUROC")
    ax.legend(frameon=False, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title("Figure S1. Era sensitivity: all years vs. \u22652020 (MGB\u2192external).",
                 fontsize=10, weight="bold", color=INK)
    plt.tight_layout()
    save("FigureS1_era_sensitivity")

# ═══════════════════════════════════════════════════════════════════════════
# S3 — mucoid sensitivity
# CONTRACT: core conclusion: mucoid-subgroup AUROCs are markedly lower for
#   MEM/CIP/FEP (out-of-domain mechanism), with task-specific boundaries.
#   Role: mechanism evidence. Archetype: quantitative grid (grouped bars).
# ═══════════════════════════════════════════════════════════════════════════
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
    ax.legend(frameon=False, fontsize=7.5)
    ax.invert_yaxis()
    ax.set_title("Figure S2. Mucoid sensitivity (Stanford only; UTSW lacks a mucoid cohort).",
                 fontsize=10, weight="bold", color=INK)
    plt.tight_layout()
    save("FigureS2_mucoid_sensitivity")

# ═══════════════════════════════════════════════════════════════════════════
# S4 — CI forest
# CONTRACT: core conclusion: all 30 transfer directions show substantial gaps;
#   color encodes training source (slots 1-3). Role: full results traceability.
#   Archetype: quantitative grid (forest plot; patient-level bootstrap 200).
# ═══════════════════════════════════════════════════════════════════════════
def fig_s4():
    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
    ci = pd.read_csv(os.path.join(R3, "transfer_ci_patient.csv"))   # TRUE patient-level CI, all 30
    # 2026-09-21：改为**固定顺序**（作者要求，原先按组内 gap 大小排列）。
    # 任务序按正文口径 MEM→CIP→LVX→CAZ→FEP，任务内六个 route 按 Table 2 的行序固定，
    # 使图与 Table 2 可以对着读。
    TASK_ORDER = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
    ROUTE_ORDER = [("MGB", "Stanford"), ("MGB", "UTSW"), ("Stanford", "MGB"),
                   ("Stanford", "UTSW"), ("UTSW", "MGB"), ("UTSW", "Stanford")]
    tm["_t"] = tm["task"].map({t: i for i, t in enumerate(TASK_ORDER)})
    tm["_r"] = [ROUTE_ORDER.index((s, g)) for s, g in zip(tm["src"], tm["tgt"])]
    tm = tm.sort_values(["_t", "_r"])
    src_col = {"MGB": S1, "Stanford": S2, "UTSW": S3}
    fig, ax = plt.subplots(figsize=(10.5, 8.6)); hairline(ax)
    # 2026-09-21：按抗生素分组（吸收 v5 版优点）。每组前一条加粗组标题、各占一个 y 位，
    # 组间以深色分隔线断开；组标题走 y 轴刻度而非独立 text()，从根上避免组头压数据行。
    ylab, ypos, seps, hdr = [], [], [], set()
    k = 0
    for task, sub in tm.groupby("task", sort=False):
        hdr.add(k); ylab.append(task.upper()); seps.append(k - 0.5); k += 1
        for _, r in sub.iterrows():
            ypos.append((k, r)); ylab.append(r["src"] + "\u2192" + r["tgt"]); k += 1
    for kk, r in ypos:
        c = ci[(ci["task"] == r["task"]) & (ci["src"] == r["src"]) & (ci["tgt"] == r["tgt"])]
        if len(c) > 0 and pd.notna(c["ci_lo"].iloc[0]):
            lo, hi = c["ci_lo"].iloc[0], c["ci_hi"].iloc[0]
        else:
            lo, hi = r["auroc"] - 0.03, r["auroc"] + 0.03
        ax.errorbar(r["auroc"], kk,
                    xerr=[[max(0, r["auroc"] - lo)], [max(0, hi - r["auroc"])]],
                    fmt="o", color=src_col.get(r["src"], INK_SEC),
                    capsize=2.2, markersize=4.5, lw=1.0, zorder=4)
        ax.axhline(y=kk, color=GRID, lw=0.4, zorder=1)
    # 2026-09-21：移植 v5 版的"内参带"——五个任务的 MGB 内部 CV AUROC 区间（§4.2 的参考水平）
    _int = sorted(tm["internal_auroc"].unique())
    INT_LO, INT_HI = float(min(_int)), float(max(_int))
    ax.axvspan(INT_LO, INT_HI, color=S1, alpha=0.08, zorder=0)
    ax.text((INT_LO + INT_HI) / 2, -0.80, "Internal CV\nreference",
            ha="center", va="center", fontsize=6.8, color=INK_MUTED,
            linespacing=1.2, zorder=5)
    ax.set_yticks(range(len(ylab))); ax.set_yticklabels(ylab, fontsize=6.4)
    for t, i2 in zip(ax.get_yticklabels(), range(len(ylab))):
        if i2 in hdr:
            t.set_fontweight("bold"); t.set_fontsize(7.0); t.set_color(INK)
    for y in seps:
        ax.axhline(y=y, color=INK_MUTED, lw=0.9, zorder=3)
    ax.set_ylim(len(ylab) - 0.5, -1.15)
    ax.set_xlabel("AUROC")
    handles = [plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S1, markersize=6, label="Source: MGB"),
               plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S2, markersize=6, label="Source: Stanford"),
               plt.Line2D([0],[0], marker="o", color="w", markerfacecolor=S3, markersize=6, label="Source: UTSW")]
    # 2026-09-21：图例移出绘图区，置于图题下方居中横排——图内无空白角落
    # （右下压 CEFEPIME 组 CI；左上压 MEROPENEM 组最左的两条 CI）。与 S1/S3 做法一致。
    fig.legend(handles=handles, loc="upper center", ncol=3, frameon=False,
               fontsize=7.5, bbox_to_anchor=(0.50, 1.012))
    fig.suptitle("Figure S4. AUROC with 95% CIs, all 30 directions (patient-level bootstrap, 200).",
                 fontsize=10, weight="bold", color=INK, y=1.055)
    plt.tight_layout()
    save("FigureS4_ci_matrix")


if __name__ == "__main__":
    print("Fig1"); fig1()
    print("Fig2"); fig2()
    print("Fig3"); fig3()
    print("S1");  fig_s1()
    print("S2");  fig_s2()
    print("S3");  fig_s3()
    print("S4");  fig_s4()
    print("Done:", OUT)
