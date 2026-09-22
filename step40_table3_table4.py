# -*- coding: utf-8 -*-
"""
Step 40 — Table 3 (偏移分解) + Table 4 (假设三: 稳定子集 vs 全特征)
从 final results 生成, 数字全部可追溯
输出: paper/tables/Table3_formatted.txt + Table4_formatted.txt + CSV
"""
import os
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
R3 = os.path.join(BASE, "05_源数据", "phase3")
OUT = os.path.join(BASE, "03_表格")
os.makedirs(OUT, exist_ok=True)

# ========== Table 3: Shift Decomposition ==========
sd = pd.read_csv(os.path.join(R3, "shift_decomposition.csv"))
# Columns: task, tgt, prev_src, prev_tgt, label_shift_pp, base_auc, ipw_auc,
# concept_residual, covariate_auc, brier_raw, brier_label_adj,
# calib_int_raw, calib_int_label_adj, top_smd

print("=" * 90)
print("Table 3. Shift Decomposition (MGB → external sites, primary directions)")
print("=" * 90)
header = f"{'Task':12s} {'Dir':8s} {'Label Δ(pp)':11s} {'Src→Tgt R-rate':16s} {'Cov AUC (no ADI)':16s} {'Brier':7s} {'Brier(LS)':9s} {'Calib Int':9s} {'Calib Int(LS)':12s} {'Top SMD (features)':30s}"
print(header)
print("-" * len(header))

for _, r in sd.iterrows():
    line = (f"{r['task'].upper():12s} {r['tgt']:8s} {r['label_shift_pp']:+.1f} pp     "
            f"{r['prev_src']:.3f} → {r['prev_tgt']:.3f}       "
            f"{r['covariate_auc']:.3f}            "
            f"{r['brier_raw']:.3f}  {r['brier_label_adj']:.3f}    "
            f"{r['calib_int_raw']:+.2f}      {r['calib_int_label_adj']:+.2f}        "
            f"{r['top_smd'][:60]}")
    print(line)

# Save formatted
with open(os.path.join(OUT, "Table3_formatted.txt"), "w", encoding="utf-8") as f:
    f.write("""================================================================================
Table 3. Label-shift, covariate-shift, and residual transfer-gap diagnostics
         (MGB → external sites, 5 primary tasks).
================================================================================

Task         Site      Label     Source→Target   Covariate  Brier   Brier   Calib.   Calib.    Top SMD features
                       Shift     Resistance      AUC        (raw)   (LS     Int.     Int.     (3 per direction)
                       (pp)      Prevalence               corr.)   (raw)    (LS
                                                                            corr.)
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
""")
    for _, r in sd.iterrows():
        f.write(f" {r['task'].upper():12s} {r['tgt']:8s}  {r['label_shift_pp']:+.1f}      {r['prev_src']:.3f} → {r['prev_tgt']:.3f}      {r['covariate_auc']:.3f}      {r['brier_raw']:.3f}   {r['brier_label_adj']:.3f}    {r['calib_int_raw']:+.2f}      {r['calib_int_label_adj']:+.2f}       {r['top_smd'][:70]}\n")
    f.write("""
──────────────────────────────────────────────────────────────────────────────────────────────────────────────────
Notes:
- Covariate AUC: two-sample classifier separability (all features, ADI excluded;
  with ADI was artifactually inflated to 0.998–0.999). Lower = less separable.
- LS corr. = Label-shift-corrected (logit-translation by prevalence ratio).
- Calib. Int. = calibration intercept (negative = overestimation of resistance).
- IPW = inverse probability weighting by density-ratio weights, truncated at
  0.05–20 (untruncated P95 weights 16,000–820,000, effective sample size
  collapsed to 5–36%; see Results §4.9).
- Top SMD: standardized mean differences, dominant features (absolute value).
- Label shift (pp) is computed from unrounded prevalences; the prevalences shown
  here are rounded to three decimals, so the printed difference may differ by 0.1 pp.
  Calibration intercepts are displayed to two decimals; the raw column matches the
  same quantity in Table 2 at that precision.
- Full decomposition table (10 rows) covers the ten primary task–site
  directions (MGB→Stanford and MGB→UTSW).
- Per-direction |SHAP| ratio medians, all-feature basis: MEM→S 0.91,
  MEM→U 0.90, CIP→S 0.82, CIP→U 0.93, LVX→S 0.95, LVX→U 1.04, CAZ→S 0.89,
  CAZ→U 0.90, FEP→S 0.83, FEP→U 0.88
  (results/phase3/feature_drift.csv).
================================================================================
""")

print("\nTable 3 saved.")

# ========== Table 4: stable-feature subset ==========
ss = pd.read_csv(os.path.join(R3, "stable_subset.csv"))
# 2026-09-18：CI 改用患者级 bootstrap（原 boot_ci 为按行重抽，未按患者聚类，区间偏窄约一半）
_ci = pd.read_csv(os.path.join(R3, "transfer_ci_patient.csv"))
_PCI = {(r["task"], r["tgt"]): f"{r['ci_lo']:.3f}–{r['ci_hi']:.3f}"
        for _, r in _ci[_ci["src"] == "MGB"].iterrows()}

print("\n" + "=" * 90)
print("Table 4. Stable-Feature Subset vs. Full-Feature Model")
print("=" * 90)
header2 = f"{'Task':12s} {'Dir':12s} {'Full Ext':9s} {'Full Gap':9s} {'Subset Ext':10s} {'Subset Gap':10s} {'Gap Δ':7s} {'95% CI (Full)':16s} {'Internal Full→Subset':20s}"
print(header2)
print("-" * len(header2))

with open(os.path.join(OUT, "Table4_formatted.txt"), "w", encoding="utf-8") as f:
    f.write("""================================================================================
Table 4. Stable-feature subset vs. full features.
================================================================================

Task         Direction    Full      Full     Subset    Subset   Gap     95% CI         Internal
                          AUROC     Gap      AUROC     Gap      Change  (Full)         Full→Subset
──────────────────────────────────────────────────────────────────────────────────────────────────
""")
    for _, r in ss.iterrows():
        line = (f" {r['task'].upper():12s} {r['tgt']:12s} {r['full_external']:.3f}    {r['full_gap']:+.3f}    "
                f"{r['subset_external']:.3f}     {r['subset_gap']:+.3f}    {r['gap_reduction']:+.3f}   "
                f"{_PCI.get((r['task'], r['tgt']), r['boot_ci']):16s}   "
                f"{r['full_internal']:.3f} → {r['subset_internal']:.3f}")
        print(line)
        f.write(line + "\n")
    f.write("""
──────────────────────────────────────────────────────────────────────────────────────────────────
Notes:
- Stable subset: top-10 features per task ranked by MGB-internal cross-setting
  SHAP stability (Spearman ρ + sign consistency). Selection uses only MGB data.
- Gap Change = (Full Gap) − (Subset Gap). Negative = gap narrowed (moved toward
  zero) by subset.
- 95% CI: patient-level bootstrap, 200 iterations, for the full-feature model.
- Tasks with large gaps (MEM, CAZ, FEP): subset narrows transfer gaps (0.02–0.10)
  at modest internal cost (0.005–0.024). Small-gap tasks (CIP, LVX): subset harms
  transfer (+0.002 to +0.067). Full discussion in Results §4.11.
================================================================================
""")

print("\nTables saved to", OUT)
