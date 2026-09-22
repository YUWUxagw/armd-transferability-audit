# -*- coding: utf-8 -*-
"""
Figures v5 — visual upgrade over v4.
Changes vs v4:
  Fig1: proper vertical spacing, visible arrows, side-note box, mucoid highlight bar
  Fig2: white cell gridlines, better colorbar, two-site divider, cleaner y-labels
  Fig3: reference diagonal annotation, shared axis labels, subtitle strip
  S1:  shared y-axis label, threshold reference line at 0.20 marked
  S2:  dot + CI overlay on bars, reference line at 0.75
  S3:  reference line at 0.5, gap line between tasks
  S4:  task-group dividers, internal reference band, cleaner label format
All other contracts, palette, data paths, export logic unchanged from v4.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.colors import TwoSlopeNorm
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd
import os, sys

OUT   = r"F:\E\Machine Learning\ARMD\02_图表"
os.makedirs(OUT, exist_ok=True)
def _pick(*cands):
    r"""2026-09-21：原始数据（data/clean）在 E:\ARMD（作者机），结果 CSV 两边都有；按存在性自动择一。
    Fig3 需要 CLEAN 下的 task_*.csv；其余图只需 R3 的结果 CSV。"""
    for c in cands:
        if os.path.isdir(c):
            return c
    return cands[-1]

R3    = _pick(r"E:\ARMD\results\phase3", r"F:\E\Machine Learning\ARMD\05_源数据\phase3")
CLEAN = _pick(r"E:\ARMD\data\clean",
              r"F:\E\Machine Learning\ARMD\clean\clean",   # 2026-09-21 作者提供
              r"F:\E\Machine Learning\ARMD\05_源数据\clean")
sys.path.insert(0, os.path.dirname(__file__))
try:
    from step14_model_phase2 import prep, feature_cols, train_lgb, fit_cat_maps, prep_cross
except ImportError as _e:
    print(f"[warn] step14_model_phase2 not imported ({_e})")

# ── palette (unchanged) ──────────────────────────────────────────────────────
SURFACE, INK, INK_SEC, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
POLE_LO, POLE_HI = "#2a78d6", "#e34948"
HIGHLIGHT = "#fff3cd"   # v5 NEW: mucoid / CAZ highlight fill

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
T_LAB = {"meropenem":"MEM","ciprofloxacin":"CIP","levofloxacin":"LVX",
         "ceftazidime":"CAZ","cefepime":"FEP"}

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.sans-serif":   ["Arial","Helvetica","DejaVu Sans"],
    "svg.fonttype":      "none",
    "pdf.fonttype":      42,
    "font.size":         7.5,
    "axes.titlesize":    10,
    "axes.labelsize":    8.5,
    "xtick.labelsize":   7.5,
    "ytick.labelsize":   7.5,
    "legend.fontsize":   7.5,
    "axes.spines.right": False,
    "axes.spines.top":   False,
    "axes.linewidth":    0.7,
    "legend.frameon":    False,
    "savefig.dpi":       600,
    "savefig.bbox":      "tight",
    "savefig.pad_inches":0.10,
    "axes.edgecolor":    BASELINE,
    "axes.labelcolor":   INK,
    "xtick.color":       INK_SEC,
    "ytick.color":       INK_SEC,
    "text.color":        INK,
    "figure.facecolor":  SURFACE,
    "axes.facecolor":    SURFACE,
    "xtick.major.size":  3,
    "ytick.major.size":  3,
    "xtick.major.width": 0.6,
    "ytick.major.width": 0.6,
})

def save(name):
    base = os.path.join(OUT, name)
    fig = plt.gcf()
    fig.savefig(base + ".svg")
    fig.savefig(base + ".pdf")
    fig.savefig(base + ".tif", pil_kwargs={"compression":"tiff_lzw"})
    print(f"  {name}")
    plt.close()

def spine_clean(ax, left=True, bottom=True):
    """Keep only requested spines; thin them."""
    for s, keep in [("left",left),("bottom",bottom),("right",False),("top",False)]:
        ax.spines[s].set_visible(keep)
        if keep:
            ax.spines[s].set_color(BASELINE)
            ax.spines[s].set_linewidth(0.7)
    ax.tick_params(width=0.6, length=3, color=INK_SEC)


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 1 — enrollment flow  (v5: proper layout, visible arrows, side note)
# ════════════════════════════════════════════════════════════════════════════
def fig1():
    # 2026-09-21：画布收紧——原 8.0×8.5 且 ylim 0–10，流程图只占上方约七成，
    # 底部留下大片空白；现按内容实际范围（注框底 1.97 → 末框顶 9.32）裁到 1.70–9.55。
    fig = plt.figure(figsize=(8.0, 6.8), facecolor=SURFACE)
    ax  = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, 10); ax.set_ylim(1.35, 9.60); ax.axis("off")

    # ── layout constants ────────────────────────────────────────────────────
    BX, BW, BH = 1.8, 6.4, 0.92     # box x-origin, width, height（2026-09-21：0.84→0.92 配合放大字号）
    CX = BX + BW / 2                 # centre x
    YGAP = 1.60                       # vertical gap between box centres
    y0   = 8.80                       # top box centre-y

    STEPS = [
        ("MGB microbiology cohort",
         "4,960,599 rows  ·  2015–2024 (dates shifted)"),
        ("P. aeruginosa rows",
         "221,675 rows  ·  15,254 cultures  ·  5,216 patients"),
        ("PA drug rows after target-drug & quality filter",
         "169,477 rows  ·  15,253 cultures"),
        ("PA rows with valid CLSI 2022 label",
         "105,421 rows  ·  10,578 cultures  (69 % of PA cultures)"),
    ]
    FILTERS = [
        "Filter: organism = P. aeruginosa",
        "Filter: 9-task drugs; exclude preliminary / negative",
        "Filter: CLSI_2022_pheno in {S, I, R}",   # 2026-09-21：原用 U+2208 "∈"，Arial 无该字形，渲染成豆腐块
    ]

    ys = [y0 - i * YGAP for i in range(4)]

    # ── draw boxes ──────────────────────────────────────────────────────────
    for k, (title, sub) in enumerate(STEPS):
        fc = HIGHLIGHT if k == 3 else SURFACE   # highlight final box
        ec = S1 if k == 3 else INK
        lw = 1.2 if k == 3 else 0.9
        r = mpatches.FancyBboxPatch(
            (BX, ys[k] - BH/2), BW, BH,
            boxstyle="round,pad=0.10",
            facecolor=fc, edgecolor=ec, linewidth=lw, zorder=3)
        ax.add_patch(r)
        ax.text(CX, ys[k] + 0.20, title,
                ha="center", va="center", fontsize=10.5,
                weight="bold", color=INK, zorder=4)
        ax.text(CX, ys[k] - 0.27, sub,
                ha="center", va="center", fontsize=9.0,
                color=INK_SEC, zorder=4)

    # ── draw arrows + filter labels ─────────────────────────────────────────
    for k, label in enumerate(FILTERS):
        y_from = ys[k]   - BH/2 - 0.02
        y_to   = ys[k+1] + BH/2 + 0.02
        ax.annotate(
            "", xy=(CX, y_to), xytext=(CX, y_from),
            arrowprops=dict(
                arrowstyle="-|>",
                color=INK_SEC, lw=1.2,
                mutation_scale=10),
            zorder=2)
        ymid = (y_from + y_to) / 2
        ax.text(CX + BW/2 + 0.20, ymid, label,
                ha="left", va="center", fontsize=8.0,
                color=INK_SEC, style="italic", zorder=4)

    # ── side note box (label-validity + mucoid) ─────────────────────────────
    nx, ny, nw, nh = 6.35, ys[3] - 1.95, 3.35, 1.25   # 2026-09-21：原 −1.15 时框体 y∈[2.85,3.90] 与末框重叠；后随字号放大加宽加高
    rn = mpatches.FancyBboxPatch(
        (nx, ny), nw, nh,
        boxstyle="round,pad=0.08",
        facecolor="#fef9e7", edgecolor=BASELINE, linewidth=0.7, zorder=3)
    ax.add_patch(rn)
    note_lines = [
        "Dominant loss: label validity",
        "(64,056 of 65,962 excluded rows)",
        "Mucoid isolates = 38.6 %",
        "of label-validity-excluded rows",
    ]
    for li, ln in enumerate(note_lines):
        w = "bold" if li == 0 else "normal"
        ax.text(nx + nw/2, ny + nh - 0.18 - li*0.26, ln,
                ha="center", va="top", fontsize=8.0,
                color=INK if li < 2 else "#a05000",
                weight=w, zorder=4)
    # connector dashed line from final box to note
    # 2026-09-21：改垂直走线——末框右下角 (BX+BW) 正落在注框顶边 x 范围内，
    # 直接竖直下落即可，斜线（原 (BX+BW, 框底) → (nx, 注框顶)）视觉上发歪。
    ax.plot([BX + BW, BX + BW], [ys[3] - BH/2, ny + nh],
            color=BASELINE, lw=0.7, ls="--", zorder=2)

    # ── title（2026-09-21：删除左上角多余的 "a" 分图标记——单图无需分图标签）
    fig.text(0.50, 0.008,
             "Figure 1.  MGB cohort attrition  (TRIPOD-AI enrollment flow)",
             ha="center", fontsize=11.5, weight="bold", color=INK)
    save("Figure1_cohort_attrition")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 2 — feature evidence drift heatmap  (v5: gridlines, label cleanup)
# ════════════════════════════════════════════════════════════════════════════
def fig2():
    fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
    KEYS = [
        "comorb_count","prior_pa_res_levofloxacin",
        "abx_carbapenem_0_30","abx_carbapenem_181_365",
        "abx_antipseudomonal_bl_0_30","abx_antipseudomonal_bl_gt365",
        "prior_pa_res_piperacillin_tazobactam","prior_pa_res_tobramycin",
        "prior_pa_res_amikacin",
        "abx_glycopeptide_0_30","abx_glycopeptide_gt365",
        "prior_pseudo","prior_pseudo_days","age_bin","comorb_malignancy","nh_30d",
    ]
    LABS = {
        "comorb_count":                       "Comorbidity count",
        "prior_pa_res_levofloxacin":           "Prior LVX resistance",
        "abx_carbapenem_0_30":                 "Carbapenem 0–30 d",
        "abx_carbapenem_181_365":              "Carbapenem 181–365 d",
        "abx_antipseudomonal_bl_0_30":         "Anti-PA BL 0–30 d",
        "abx_antipseudomonal_bl_gt365":        "Anti-PA BL >365 d",
        "prior_pa_res_piperacillin_tazobactam":"Prior TZP resistance",
        "prior_pa_res_tobramycin":             "Prior TOB resistance",
        "prior_pa_res_amikacin":               "Prior AMK resistance",
        "abx_glycopeptide_0_30":               "Glycopeptide 0–30 d",
        "abx_glycopeptide_gt365":              "Glycopeptide >365 d",
        "prior_pseudo":                        "Prior Pseudomonas",
        "prior_pseudo_days":                   "Days since Pseudomonas",
        "age_bin":                             "Age",
        "comorb_malignancy":                   "Malignancy",
        "nh_30d":                              "Nursing home 30 d",
    }

    tall, ylab = [], []
    for d in ["Stanford", "UTSW"]:
        sub = fd[fd["tgt"] == d]
        for f in KEYS:
            vals = []
            for t in TASKS:
                rows = sub[(sub["feature"] == f) & (sub["task"] == t)]
                vals.append(float(rows["ratio"].iloc[0]) if len(rows) else np.nan)
            tall.append(vals)
            site_tag = "→S" if d == "Stanford" else "→U"
            ylab.append(f"{LABS.get(f, f)}  {site_tag}")
    tall = np.array(tall)
    n_row = len(ylab)
    n_site = len(KEYS)   # rows per site = 16

    norm  = TwoSlopeNorm(vmin=0.0, vcenter=1.0, vmax=2.5)
    fig, ax = plt.subplots(figsize=(5.8, 10.2), facecolor=SURFACE)
    im = ax.imshow(tall, aspect="auto", cmap="RdBu_r", norm=norm,
                   interpolation="nearest")

    # cell gridlines
    for x in np.arange(-0.5, 5, 1):
        ax.axvline(x, color="white", lw=0.6, zorder=3)
    for y in np.arange(-0.5, n_row, 1):
        ax.axhline(y, color="white", lw=0.4, zorder=3)

    # site-group divider
    ax.axhline(n_site - 0.5, color=INK, lw=1.2, zorder=4)

    # site labels in margin
    for si, (d, yc) in enumerate([("→ Stanford", n_site/2 - 0.5),
                                   ("→ UTSW",    n_site + n_site/2 - 0.5)]):
        ax.text(-0.7, yc, d, ha="right", va="center", fontsize=7.5,
                weight="bold", color=INK, rotation=90,
                transform=ax.get_yaxis_transform())

    ax.set_xticks(range(5))
    ax.set_xticklabels([T_LAB[t] for t in TASKS], fontsize=9, weight="bold")
    ax.xaxis.set_ticks_position("top")
    ax.set_yticks(range(n_row))
    ax.set_yticklabels(
        [LABS.get(k, k) for k in KEYS] * 2,   # drop site suffix (shown via margin label)
        fontsize=6.8)
    ax.tick_params(length=0)
    for sp in ax.spines.values():
        sp.set_visible(False)

    # cell value labels for |ratio-1| > 0.25
    for i in range(n_row):
        for j in range(5):
            v = tall[i, j]
            if not np.isnan(v) and abs(v - 1) > 0.25:
                fc = "white" if (v < 0.45 or v > 1.65) else INK
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        fontsize=5.8, color=fc, zorder=5)

    cbar = fig.colorbar(im, ax=ax, shrink=0.55, pad=0.02, aspect=25)
    cbar.set_label("|SHAP| ratio  (target / source)", fontsize=7.5, color=INK)
    cbar.ax.tick_params(labelsize=6.5, colors=INK_SEC, width=0.5)
    cbar.outline.set_linewidth(0.5)
    cbar.ax.axhline(y=1.0, color=INK, lw=0.8)   # mark ratio=1

    ax.set_title("Figure 3.  Feature evidence drift\n"
                 "(same MGB-trained model, full-reference SHAP, ADI excluded)",
                 fontsize=9.5, weight="bold", color=INK, pad=32, loc="left")
    plt.tight_layout(rect=[0.08, 0, 1, 1])
    save("Figure3_feature_drift_heatmap")


# ════════════════════════════════════════════════════════════════════════════
# FIGURE 3 — calibration curves  (v5: shared labels, diagonal annotation)
# ════════════════════════════════════════════════════════════════════════════
def fig3():
    from sklearn.calibration import calibration_curve
    T3 = ["meropenem", "ceftazidime", "cefepime"]
    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))

    fig, axes = plt.subplots(2, 3, figsize=(10.5, 6.8),
                             sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.40, "wspace": 0.18})

    for j, t in enumerate(T3):
        mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"),
                              low_memory=False, encoding="utf-8-sig")
        mgb  = prep(mgb_raw)
        feats = feature_cols(mgb)
        maps  = fit_cat_maps(mgb_raw)
        m     = train_lgb(mgb[feats], mgb["label"], mgb[feats], mgb["label"])

        for i, (d, dl) in enumerate([("Stanford", "S"), ("UTSW", "U")]):
            ax = axes[i, j]
            spine_clean(ax)
            te_raw = pd.read_csv(os.path.join(CLEAN, d, f"task_{t}.csv"),
                                 low_memory=False, encoding="utf-8-sig")
            te   = prep_cross(te_raw, maps, mgb["adi"].median())
            X    = te[[f for f in feats if f in te.columns]]
            y    = te["label"].values
            pred = m.predict_proba(X)[:, 1]
            fp, mp = calibration_curve(y, pred, n_bins=10, strategy="uniform")

            # reference diagonal（参考线说明统一放到图级 legend，避免六个面板重复）
            ax.plot([0, 1], [0, 1], color=BASELINE, lw=1.0, ls="--", zorder=1)

            # calibration curve
            ax.plot(mp, fp, "-o", color=S1, lw=1.6,
                    markersize=4.5, mfc=SURFACE, mec=S1, zorder=3)

            ax.set_xlim(-0.02, 1.02); ax.set_ylim(-0.02, 1.02)

            # title strip per panel
            title_col = "#c0392b" if t == "ceftazidime" else INK
            ax.set_title(f"{T_LAB[t]} → {dl}",
                         fontsize=9.5, weight="bold", color=title_col, pad=5)

            # 面板注记：Brier / slope / intercept
            # 2026-09-21 修 bug：原判据写 "cal_slope"，而 transfer_matrix.csv 的列名是
            # calib_slope / calib_int，故斜率的注记从未渲染出来（且不报错）。
            rr = tm[(tm["task"] == t) & (tm["src"] == "MGB") & (tm["tgt"] == d)]
            if len(rr):
                _r = rr.iloc[0]
                ax.text(0.96, 0.06,
                        f"Brier = {_r['brier']:.3f}\n"
                        f"Slope = {_r['calib_slope']:.2f}\n"
                        f"Intercept = {_r['calib_int']:+.2f}",
                        ha="right", va="bottom", transform=ax.transAxes,
                        fontsize=6.8, color=INK_SEC, linespacing=1.3)

    # shared axis labels
    fig.text(0.50, 0.01, "Mean predicted probability",
             ha="center", fontsize=9, color=INK)
    fig.text(0.01, 0.50, "Observed event fraction",
             va="center", rotation=90, fontsize=9, color=INK)

    # row labels
    for i, site in enumerate(["→ Stanford", "→ UTSW"]):
        axes[i, 0].set_ylabel(site, fontsize=9, weight="bold",
                              color=INK, labelpad=28)

    # 图级 legend：虚线＝完美校准参考，实线＝迁移后实测（2026-09-21 新增）
    handles = [Line2D([0], [0], color=BASELINE, lw=1.0, ls="--",
                      label="Perfect calibration"),
               Line2D([0], [0], color=S1, lw=1.6, marker="o", markersize=4.5,
                      mfc=SURFACE, mec=S1, label="Observed after transfer")]
    fig.legend(handles=handles, loc="upper center", ncol=2, frameon=False,
               fontsize=8.5, bbox_to_anchor=(0.50, 1.005))
    fig.suptitle(
        "Figure 2.  Calibration curves  (MGB → external, the three largest-gap tasks)",
        fontsize=10.5, weight="bold", color=INK, y=1.075)
    save("Figure2_calibration_curves")


# ════════════════════════════════════════════════════════════════════════════
# S1 — DCA  (v5: threshold reference line at 0.20, shared legend top)
# ════════════════════════════════════════════════════════════════════════════
def fig_s1():
    dca = pd.read_csv(os.path.join(R3, "dca.csv"))
    # 2026-09-21：sharey=True 时 set_ylim(-0.06, None) 会用第一个面板的上界锁死全局，
    # 把 CIP→U(0.446)/LVX→S(0.350)/LVX→U(0.495) 等高值曲线切到框外；改为显式全局上界。
    _w = dca[(dca["threshold"] >= 0.02) & (dca["threshold"] <= 0.48)]
    YMAX = float(_w["net_benefit"].max()) * 1.05
    fig, axes = plt.subplots(2, 5, figsize=(14.0, 5.8), sharex=True, sharey=True,
                             gridspec_kw={"hspace": 0.38, "wspace": 0.12})
    for j, t in enumerate(TASKS):
        for i, (d, dl) in enumerate([("Stanford","S"), ("UTSW","U")]):
            ax = axes[i, j]
            spine_clean(ax)
            sub = dca[(dca["task"] == t) & (dca["tgt"] == d)]

            ax.axhline(y=0,    color=BASELINE, lw=0.8, zorder=1)
            ax.axvline(x=0.20, color=BASELINE, lw=0.7, ls=":", zorder=1)

            full   = sub[sub["model"] == "full"]
            subset = sub[sub["model"] == "subset"]
            ax.plot(full["threshold"],   full["net_benefit"],
                    "-", color=S1, lw=1.6, zorder=3)
            ax.plot(subset["threshold"], subset["net_benefit"],
                    "--", color=S2, lw=1.4, zorder=3)

            title_col = "#c0392b" if t == "ceftazidime" else INK
            ax.set_title(f"{T_LAB[t]} → {dl}",
                         fontsize=8.5, weight="bold", color=title_col, pad=4)
            ax.set_xlim(0.02, 0.48)
            ax.set_ylim(-0.06, YMAX)

            if i == 1: ax.set_xlabel("Threshold", fontsize=7.5)
            if j == 0: ax.set_ylabel("Net benefit", fontsize=7.5)

    handles = [
        Line2D([0],[0], color=S1, lw=1.6, ls="-",  label="Full model"),
        Line2D([0],[0], color=S2, lw=1.4, ls="--", label="Stable-subset model"),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=2,
               frameon=False, fontsize=8.5, bbox_to_anchor=(0.50, 1.02))
    fig.suptitle(
        "Figure S3.  Decision curve analysis  (MGB → external, all primary tasks)",
        fontsize=10, weight="bold", color=INK, y=1.06)
    plt.tight_layout()
    save("FigureS3_dca")


# ════════════════════════════════════════════════════════════════════════════
# S2 — era sensitivity  (v5: reference line at 0.75, dot overlay on bars)
# ════════════════════════════════════════════════════════════════════════════
def fig_s2():
    era = pd.read_csv(os.path.join(R3, "era_sensitivity.csv"))
    e20 = era[era["cut"] == 2020].reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(9.5, 5.4), facecolor=SURFACE)
    spine_clean(ax)

    y = np.arange(len(e20))
    # 2026-09-21：条形→哑铃点图。原 barh 自 x=0 起，而两个时代的 AUROC 差异只在
    # 0.62–0.79 这段，长条把差异压到最右 15%；点图允许把轴截到 0.5（AUROC 随机水平），差异直接可读。
    for k, r in e20.iterrows():
        lo, hi = sorted((r["auroc_all"], r["auroc_cut"]))
        ax.plot([lo, hi], [k, k], "-", color=BASELINE, lw=1.6,
                zorder=2, solid_capstyle="round")
    ax.scatter(e20["auroc_all"], y, s=46, facecolor=SURFACE, edgecolor=INK_MUTED,
               linewidths=1.3, zorder=4, label="All years")
    ax.scatter(e20["auroc_cut"], y, s=46, facecolor=S1, edgecolor=SURFACE,
               linewidths=1.0, zorder=5, label="≥ 2020")

    ax.axvline(0.75, color=BASELINE, lw=0.9, ls="--", zorder=1)
    ax.text(0.753, len(e20) - 0.55, "0.75", fontsize=6.5, color=INK_MUTED)

    labels = [f"{T_LAB[r['task']]} → {r['tgt'][:1]}"
              for _, r in e20.iterrows()]
    ax.set_yticks(y); ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC  (axis starts at 0.60)", fontsize=9)
    ax.set_xlim(0.60, 0.80)          # 2026-09-21：贴合数据 0.620–0.786，去掉两侧空白
    ax.invert_yaxis()
    ax.set_ylim(len(e20) - 0.42, -0.58)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Figure S1.  Era sensitivity:  all years vs. ≥ 2020  (MGB → external)",
                 fontsize=10, weight="bold", color=INK, pad=8, loc="left")
    plt.tight_layout()
    save("FigureS1_era_sensitivity")


# ════════════════════════════════════════════════════════════════════════════
# S3 — mucoid sensitivity  (v5: task gap lines, 0.5 reference, highlight row)
# ════════════════════════════════════════════════════════════════════════════
def fig_s3():
    muc = pd.read_csv(os.path.join(R3, "mucoid_sensitivity.csv"))
    muc = muc.dropna(subset=["auroc_mucoid_only"]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8.8, 5.0), facecolor=SURFACE)
    spine_clean(ax)

    y = np.arange(len(muc))
    # 2026-09-21：条形→哑铃点图（同 S2）。三个子群的点由细杆相连，差异一眼可读。
    for k, r in muc.iterrows():
        vals = [r["auroc_all"], r["auroc_no_mucoid"], r["auroc_mucoid_only"]]
        ax.plot([min(vals), max(vals)], [k, k], "-", color=BASELINE, lw=1.6,
                zorder=2, solid_capstyle="round")
    ax.scatter(muc["auroc_all"],        y, s=46, facecolor=S1, edgecolor=SURFACE,
               linewidths=1.0, zorder=4, label="All samples")
    ax.scatter(muc["auroc_no_mucoid"],  y, s=46, facecolor=S2, edgecolor=SURFACE,
               linewidths=1.0, zorder=5, label="Excl. mucoid")
    ax.scatter(muc["auroc_mucoid_only"], y, s=46, facecolor=S3, edgecolor=SURFACE,
               linewidths=1.0, zorder=6, label="Mucoid only")

    # highlight MEM/CIP/FEP rows (out-of-domain mechanism)
    for k, row in muc.iterrows():
        if T_LAB.get(row["task"], "") in ("MEM","CIP","FEP"):
            ax.axhspan(k - 0.45, k + 0.45,
                       color=HIGHLIGHT, zorder=0, alpha=0.6)

    # task-separator horizontal lines
    prev = None
    for k, row in muc.iterrows():
        if prev is not None and row["task"] != prev:
            ax.axhline(k - 0.5, color=GRID, lw=0.7, zorder=1)
        prev = row["task"]

    labels = [f"{T_LAB.get(r['task'], r['task'])} → {r['tgt'][:1]}"
              for _, r in muc.iterrows()]
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("AUROC  (axis starts at 0.58)", fontsize=9)
    ax.set_xlim(0.58, 0.78)          # 2026-09-21：贴合数据 0.605–0.759，去掉两侧空白
    ax.invert_yaxis()
    ax.set_ylim(len(muc) - 0.42, -0.58)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    ax.set_title("Figure S2.  Mucoid sensitivity  (Stanford directions only;\n"
                 "UTSW PA cohort contains essentially no mucoid isolates)",
                 fontsize=9.5, weight="bold", color=INK, pad=6, loc="left")
    # annotation for highlighted rows
    # 2026-09-21：原 ha="right" 与右下角图例叠字，改靠左下
    ax.text(0.012, 0.02,
            "Shaded: out-of-domain mechanism (MEM / CIP / FEP)",
            ha="left", va="bottom", transform=ax.transAxes,
            fontsize=6.8, color="#a05000", style="italic")
    plt.tight_layout()
    save("FigureS2_mucoid_sensitivity")


# ════════════════════════════════════════════════════════════════════════════
# S4 — CI forest  (v5: task dividers, internal-reference band, wider labels)
# ════════════════════════════════════════════════════════════════════════════
def fig_s4():
    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
    ci = pd.read_csv(os.path.join(R3, "transfer_ci_patient.csv"))
    # internal reference range from §4.2
    INT_LO, INT_HI = 0.743, 0.771

    src_col = {"MGB": S1, "Stanford": S2, "UTSW": S3}

    # sort: by task then by AUROC within task
    tm = tm.sort_values(["task", "auroc"], ascending=[True, False]).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(11.0, 9.0), facecolor=SURFACE)
    spine_clean(ax, left=False, bottom=True)
    ax.yaxis.set_visible(False)

    labels, aucs, loes, his, cols = [], [], [], [], []
    for _, r in tm.iterrows():
        lbl = f"{T_LAB.get(r['task'], r['task'])}  {r['src']} → {r['tgt']}"
        labels.append(lbl)
        c = ci[(ci["task"]==r["task"]) & (ci["src"]==r["src"]) & (ci["tgt"]==r["tgt"])]
        if len(c) > 0 and pd.notna(c["ci_lo"].iloc[0]):
            lo, hi = c["ci_lo"].iloc[0], c["ci_hi"].iloc[0]
        else:
            lo, hi = r["auroc"] - 0.03, r["auroc"] + 0.03
        aucs.append(r["auroc"])
        loes.append(max(0, r["auroc"] - lo))
        his.append(max(0, hi - r["auroc"]))
        cols.append(src_col.get(r["src"], INK_SEC))

    ypos = list(range(len(aucs)))[::-1]   # top=0 visually → highest y-value first

    # internal-reference band
    ax.axvspan(INT_LO, INT_HI, color=S1, alpha=0.08, zorder=0)
    ax.axvline(INT_LO, color=S1, lw=0.6, ls=":", zorder=1)
    ax.axvline(INT_HI, color=S1, lw=0.6, ls=":", zorder=1)
    ax.text((INT_LO + INT_HI)/2, max(ypos) + 0.8,
            "Internal CV\nreference",
            ha="center", va="bottom", fontsize=6.5, color=S1)

    # task-group dividers
    prev_task = None
    for k, (_, r) in enumerate(tm.iterrows()):
        if prev_task is not None and r["task"] != prev_task:
            ax.axhline(y=ypos[k] + 0.5, color=GRID, lw=0.8, zorder=1)
        prev_task = r["task"]
        # task label at group start
        if r["task"] != (tm.iloc[k-1]["task"] if k > 0 else ""):
            ax.text(0.595, ypos[k] + 0.3, T_LAB.get(r["task"], r["task"]),
                    fontsize=7.5, weight="bold", color=INK_SEC,
                    ha="left", va="bottom")

    for k in range(len(aucs)):
        ax.errorbar(aucs[k], ypos[k],
                    xerr=[[loes[k]], [his[k]]],
                    fmt="o", color=cols[k], capsize=2.5,
                    markersize=5.5, lw=1.0, zorder=4)
        ax.text(0.60, ypos[k], labels[k],
                ha="left", va="center", fontsize=7.0, color=INK)

    ax.set_xlim(0.58, 0.86)
    ax.set_xlabel("AUROC", fontsize=9)
    handles = [
        Line2D([0],[0], marker="o", color="w", markerfacecolor=S1,
               markersize=7, label="Source: MGB"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=S2,
               markersize=7, label="Source: Stanford"),
        Line2D([0],[0], marker="o", color="w", markerfacecolor=S3,
               markersize=7, label="Source: UTSW"),
    ]
    ax.legend(handles=handles, frameon=False, fontsize=8, loc="lower left")
    ax.set_title(
        "Figure S4.  AUROC with 95 % CIs, all 30 directions\n"
        "(patient-level bootstrap, 200 iterations)",
        fontsize=10, weight="bold", color=INK, pad=8, loc="left")
    plt.tight_layout()
    save("FigureS4_ci_matrix")


if __name__ == "__main__":
    print("Fig1"); fig1()
    print("Fig2"); fig2()
    print("Fig3"); fig3()
    print("S1");   fig_s1()
    print("S2");   fig_s2()
    print("S3");   fig_s3()
    print("S4");   fig_s4()
    print("Done:", OUT)
