# -*- coding: utf-8 -*-
"""
Figures v5 — transfer-gap sensitivity forest plot (nature-figure compliant).

FIGURE CONTRACT
1. Core conclusion: the three data-level interventions (drop-I, mucoid removal,
   >=2020 era) narrow the transfer gap in most directions and bring the
   joint-intervention 95% CI to the prespecified internal reference in nine of
   ten; a single direction retains a persistent residual (CAZ->U), the only one
   whose CI lies entirely below the reference across all five bootstrap seeds.
2. Evidence chain (single panel carries the claim):
   - primary gap point (open circle) and joint-intervention gap point (filled
     square) per direction, connected by a hairline
   - joint-intervention patient-level 95% CI error bar (200 bootstrap iterations)
   - zero line = MGB internal reference; direction-significance read as
     whether the internal reference lies above the CI
   - CAZ directions emphasized (the persistent residual); the single
     seed-stable residual annotated
3. Archetype: quantitative grid (forest plot).
4. Backend: Python (matplotlib), exclusive.
5. Export contract: Arial, font 7pt, SVG editable text + PDF fonttype 42 +
   TIFF 600 dpi LZW; palette = dataviz-validated reference palette.

2026-09-18 修复：
  - 路径迁移（原硬编码 E:\\ARMD，该盘已废弃）→ 现项目目录
  - INTERNAL 由硬编码改为从 transfer_matrix.csv 读取（原硬编码为
    early-stopping 泄漏修复**前**的膨胀值，与正文内参不符）
  - 星号集合由硬编码 sig_dirs 改为从 boot_seed_sensitivity.csv 的
    sig_all_seeds 列读取（原硬编码把 MEM→U / CAZ→S / FEP→U 也标为显著，
    与图注、正文 §3.8、§4.14 及源数据三方相悖）
  - 标题去掉禁令词 "Fixable-gap"
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import os

ROOT  = r"F:\E\Machine Learning\ARMD"
OUT   = os.path.join(ROOT, "02_图表")
R3    = os.path.join(ROOT, "05_源数据", "phase3")
os.makedirs(OUT, exist_ok=True)

SURFACE, INK, INK_SEC, INK_MUTED = "#fcfcfb", "#0b0b0b", "#52514e", "#898781"
GRID, BASELINE = "#e1e0d9", "#c3c2b7"
S1, S2, S3 = "#2a78d6", "#eb6834", "#1baf7a"
CAZ_EMPH = "#e34948"

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

TASKS = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
T_LAB = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX",
         "ceftazidime": "CAZ", "cefepime": "FEP"}

# 内参：取自权威结果文件，禁止字面量
_tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
INTERNAL = {t: float(_tm[_tm["task"] == t]["internal_auroc"].iloc[0]) for t in TASKS}

# 星号集合：唯一判据 = 五个 bootstrap 种子下 CI 全部低于内参
_bs = pd.read_csv(os.path.join(R3, "boot_seed_sensitivity.csv"))
sig_dirs = {f"{T_LAB[r['task']]}→{r['tgt'][0]}" for _, r in _bs.iterrows()
            if bool(r["sig_all_seeds"])}

jc = pd.read_csv(os.path.join(R3, "joint_fix_ci.csv"))
ledger = pd.read_csv(os.path.join(R3, "attribution_ledger.csv"))

# order: task blocks, each with Stanford then UTSW
rows = []
for t in TASKS:
    for site in ["Stanford", "UTSW"]:
        j = jc[(jc["task"] == t) & (jc["tgt"] == site)].iloc[0]
        l = ledger[(ledger["task"] == t) & (ledger["tgt"] == site)].iloc[0]
        # CIs are on the AUROC scale; convert to gap scale vs the internal reference
        rows.append(dict(task=t, site=site[:1], prim_gap=l["primary_gap"],
                         joint_gap=l["fixed_gap"],
                         ci_lo=j["ci_lo"] - INTERNAL[t],
                         ci_hi=j["ci_hi"] - INTERNAL[t]))
rows = pd.DataFrame(rows)
y = np.arange(len(rows))[::-1]

fig, ax = plt.subplots(figsize=(6.4, 4.4))

# zero reference line (internal baseline)
ax.axvline(0.0, color=BASELINE, linewidth=0.9, zorder=1)

# CAZ background band (the persistent residual)
for i, r in rows.iterrows():
    if r["task"] == "ceftazidime":
        ax.axhspan(y[i] - 0.5, y[i] + 0.5, color=CAZ_EMPH, alpha=0.045, zorder=0)

for i, r in rows.iterrows():
    yi = y[i]
    is_caz = r["task"] == "ceftazidime"
    # connector: primary -> joint
    ax.plot([r["prim_gap"], r["joint_gap"]], [yi, yi], color=GRID, linewidth=0.9, zorder=2)
    # primary gap (open circle)
    ax.scatter([r["prim_gap"]], [yi], s=15, facecolor=SURFACE, edgecolor=S1,
               linewidth=0.9, zorder=3)
    # joint CI (gap scale)
    lw = 1.5 if is_caz else 1.1
    ax.plot([r["ci_lo"], r["ci_hi"]], [yi, yi], color=S3, linewidth=lw, zorder=3)
    ax.plot([r["ci_lo"] - 0.004, r["ci_lo"] + 0.004], [yi, yi], color=S3, linewidth=lw)
    ax.plot([r["ci_hi"] - 0.004, r["ci_hi"] + 0.004], [yi, yi], color=S3, linewidth=lw)
    # joint gap point (filled square)
    ax.scatter([r["joint_gap"]], [yi], s=22, marker="s", facecolor=S3,
               edgecolor="none", zorder=4)

# y labels: task-site
ylab = [f"{T_LAB[r['task']]}→{r['site']}" for _, r in rows.iterrows()]
ax.set_yticks(y)
ax.set_yticklabels(ylab, fontsize=7.5)
for i, r in rows.iterrows():
    if r["task"] == "ceftazidime":
        ax.get_yticklabels()[i].set_color(CAZ_EMPH)
        ax.get_yticklabels()[i].set_fontweight("bold")

# task separators
for t in TASKS[1:]:
    last_idx = rows.index[rows["task"] == t].max()
    ax.axhline(y[last_idx] + 0.5, color=GRID, linewidth=0.6, linestyle=(0, (1, 2)), zorder=0)

ax.set_xlabel("Transfer gap (external AUROC − internal CV AUROC)", fontsize=8.5)
ax.set_xlim(-0.24, 0.14)
ax.set_ylim(-0.8, len(rows) - 0.2)
ax.grid(axis="x", color=GRID, linewidth=0.5, zorder=0)
ax.set_axisbelow(True)

# significance annotation: the direction whose joint-intervention CI lies
# entirely below the internal reference in every one of five bootstrap seeds
for i, r in rows.iterrows():
    lab = f"{T_LAB[r['task']]}→{r['site']}"
    if lab in sig_dirs:
        ax.annotate("*", xy=(r["ci_hi"] + 0.006, y[i]), fontsize=9,
                    color=CAZ_EMPH if r["task"] == "ceftazidime" else INK_SEC, va="center")

# legend
handles = [
    plt.Line2D([0], [0], marker="o", linestyle="None", markerfacecolor=SURFACE,
               markeredgecolor=S1, markeredgewidth=0.9, markersize=5, label="Primary gap"),
    plt.Line2D([0], [0], marker="s", linestyle="None", markerfacecolor=S3,
               markersize=5, label="Joint-intervention gap"),
    plt.Line2D([0], [0], color=S3, linewidth=1.4, label="Patient-level 95% CI"),
]
leg = ax.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.012, 0.02),
                frameon=False, fontsize=6.8)
ax.text(0.99, 0.995,
        "* CI lies entirely below the internal reference in all five bootstrap seeds",
        transform=ax.transAxes, ha="right", va="top", fontsize=6.2, color=INK_MUTED)

fig.suptitle("Transfer-gap sensitivity to data-alignment interventions: "
             "primary vs. joint-intervention gaps",
             fontsize=9.5, weight="bold", color=INK, y=0.99)

base = os.path.join(OUT, "fig4_ledger_forest")
fig.savefig(base + ".svg")
fig.savefig(base + ".pdf")
fig.savefig(base + ".tif", pil_kwargs={"compression": "tiff_lzw"})
print(f"fig4_ledger_forest saved (svg/pdf/tif; tif {os.path.getsize(base + '.tif')//1024} KB)")
print("marked directions (sig_all_seeds):", sorted(sig_dirs))
print("internal reference:", {k: round(v, 6) for k, v in INTERNAL.items()})
