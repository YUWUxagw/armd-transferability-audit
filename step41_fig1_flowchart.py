# -*- coding: utf-8 -*-
"""
Step 41 — Figure 1: MGB cohort attrition flowchart
TRIPOD-AI required: enrollment flow diagram
TIFF ≥300 DPI, no color dependence, publication-ready
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

OUT = r"E:\ARMD\paper\figures"
os.makedirs(OUT, exist_ok=True)

# ── MPA 0.5 stroke-colour-only palette (colorblind-safe, grayscale-printable) ──
INK   = "#1a1a1a"
GREY  = "#4d4d4d"
BOX_FACE = "#f5f5f5"
ARROW = "#666666"

plt.rcParams.update({
    "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
    "font.size": 9, "axes.titlesize": 11, "axes.labelsize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "text.color": INK, "axes.edgecolor": GREY, "axes.labelcolor": INK,
    "xtick.color": GREY, "ytick.color": GREY,
})

fig, ax = plt.subplots(figsize=(7.5, 7.8))
ax.set_xlim(0, 10); ax.set_ylim(0, 10)
ax.axis("off")

# Box helper
def box(y, label, sub="", x0=3.0, w=4.0, h=0.65, face=BOX_FACE, edge=INK):
    r = mpatches.FancyBboxPatch((x0, y - h/2), w, h, boxstyle="round,pad=0.1",
                                 facecolor=face, edgecolor=edge, linewidth=1.2, zorder=3)
    ax.add_patch(r)
    ax.text(x0 + w/2, y, label, ha="center", va="center", fontsize=9, weight="bold", color=INK, zorder=4)
    if sub:
        ax.text(x0 + w/2, y - 0.32, sub, ha="center", va="center", fontsize=7.5, color=GREY, zorder=4)

def arrow(y_from, y_to, label="", x=5.0):
    ax.annotate("", xy=(x, y_to + 0.3), xytext=(x, y_from - 0.3),
                arrowprops=dict(arrowstyle="->", color=ARROW, lw=1.3), zorder=2)
    if label:
        ax.text(x + 2.25, (y_from + y_to)/2, label, ha="left", va="center", fontsize=7.5, color=GREY, zorder=4)

def side_note(y, text):
    ax.text(7.8, y, text, ha="left", va="center", fontsize=7.2, color=GREY, style="italic", zorder=4)

# ── Diagram ──
y = 9.2
box(y, "MGB microbiology cohort", "4,960,599 rows · 5 years (2015–2024, shifted)")
arrow(y, y-0.7, "organism starts with \"PSEUDOMONAS AERUGINOSA\"")
y -= 0.75
box(y, "P. aeruginosa rows", "221,675 rows · 15,254 unique cultures · 5,216 patients")
arrow(y, y-0.7, "drug in 9-task set & ~preliminary & ~negative culture")
y -= 0.75
box(y, "PA target-drug rows after exclusions", "169,477 rows · 15,253 cultures")
arrow(y, y-0.7, "CLSI_2022_pheno in {Susceptible, Intermediate, Resistant}")
y -= 0.75
box(y, "PA rows with valid label", "105,421 rows · 10,578 cultures (69% of PA cultures)")
side_note(y, "37.8% lost: no interpretable CLSI phenotype\n(mucoid isolates 38.6% of lost; see §9)")

fig.suptitle("Figure 1. MGB cohort attrition (TRIPOD-AI enrollment flow).", fontsize=11,
             fontweight="bold", color=INK, y=0.98)

# Sub-footnote
fig.text(0.5, 0.015, "Numbers reflect the PA task superset (≥1 of 9 target drugs tested). "
         "Per-task cohorts are derived from the superset and may be smaller.",
         ha="center", fontsize=7, color=GREY, style="italic")

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
fig.savefig(os.path.join(OUT, "Figure1_cohort_attrition.tif"), dpi=300, pil_kwargs={"compression": "tiff_lzw"})
plt.close()
print("Figure 1 saved:", os.path.join(OUT, "Figure1_cohort_attrition.tif"))
