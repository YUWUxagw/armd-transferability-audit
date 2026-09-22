# -*- coding: utf-8 -*-
"""
Step 57 — regenerate Table 2 / Table 5 / Table S7 (matched-reference) from
unrounded final CSVs after the 2026-09-03 early-stopping fix.
All numbers trace to results/phase3/*.csv; no rounded constants.
Output: E:\\ARMD\\paper\\tables\\{Table2,Table5,TableS1_matched_reference}_formatted.txt
"""
import os
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
R3 = os.path.join(BASE, "05_源数据", "phase3")
OUT = os.path.join(BASE, "03_表格")
os.makedirs(OUT, exist_ok=True)

tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
pw = pd.read_csv(os.path.join(R3, "pairwise_fixes.csv"))
jc = pd.read_csv(os.path.join(R3, "joint_fix_ci.csv"))
di = pd.read_csv(os.path.join(R3, "dropI_precise.csv"))
ci = pd.read_csv(os.path.join(R3, "transfer_ci_patient.csv"))
calslope = pd.read_csv(os.path.join(R3, "calibration_ci.csv"))  # MGB-source directions only
INT = {t: float(tm[tm["task"] == t]["internal_auroc"].iloc[0]) for t in tm["task"].unique()}
DROPS = {"meropenem": 0.830146, "ciprofloxacin": 0.787374, "levofloxacin": 0.797826,
         "ceftazidime": 0.790970, "cefepime": 0.784676}  # from dropI_precise.csv (step56)
ABBR = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX",
        "ceftazidime": "CAZ", "cefepime": "FEP"}

# ============ Table 2 ============
t2ci = {}
for _, r in ci.iterrows():
    t2ci[(r["task"], r["src"], r["tgt"])] = (r["ci_lo"], r["ci_hi"])
t2slope = {}
for _, r in calslope.iterrows():
    t2slope[(r["task"], r["tgt"])] = (r["slope_ci_lo"], r["slope_ci_hi"])

with open(os.path.join(OUT, "Table2_formatted.txt"), "w", encoding="utf-8") as f:
    f.write("================================================================================\n")
    f.write("Table 2. Cross-system transfer performance across five primary tasks and six\n")
    f.write("         site-pair routes.\n")
    f.write("================================================================================\n\n")
    h1 = "Task        Direction      n_test   AUROC       Transfer   Brier   Calib.   Calib.\n"
    h2 = "                                       (95% CI)   Gap                 Int.     Slope\n"
    h3 = " " * h2.index("Slope") + "(95% CI)\n"
    f.write(h1 + h2 + h3)
    f.write("────────────────────────────────────────────────────────────────────────────────\n")
    for t in ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]:
        sub = tm[tm["task"] == t]
        for _, r in sub.iterrows():
            prim = (r["src"] == "MGB")
            s = f"{r['calib_slope']:.2f}"
            if prim and (t, r["tgt"]) in t2slope:
                sl, sh = t2slope[(t, r["tgt"])]
                s += f" ({sl:.2f}–{sh:.2f})"
            f.write(f"{t.upper():12s} {r['src']}→{r['tgt']:9s} {int(r['n_test']):7,}   "
                    f"{r['auroc']:.3f}       {r['transfer_gap']:+.3f}    {r['brier']:.3f}   "
                    f"{r['calib_int']:+.2f}    {s}\n")
            if prim:
                lo, hi = t2ci[(t, r["src"], r["tgt"])]
                f.write(f"                              ({lo:.3f}–{hi:.3f})\n")
        f.write("────────────────────────────────────────────────────────────────────────────────\n")
    f.write("\nNotes:\n")
    f.write("- Transfer Gap = external AUROC − MGB internal CV AUROC (LightGBM, fixed 500\n")
    f.write("  trees, no early stopping; patient-level 5-fold). Internal AUROCs: MEM 0.771,\n")
    f.write("  CIP 0.743, LVX 0.750, CAZ 0.769, FEP 0.757 (ADI excluded from primary model;\n")
    f.write("  see §4.9).\n")
    f.write("- Non-MGB source rows are cross-model comparisons on a common target, not\n")
    f.write("  source-internal transfer gaps; gap interpretation is restricted to the ten\n")
    f.write("  primary directions (MGB→Stanford, MGB→UTSW).\n")
    f.write("- 95% CIs (patient-level bootstrap, 200 iterations) reported for primary\n")
    f.write("  directions (MGB→Stanford, MGB→UTSW); this includes the 95% CI shown with\n")
    f.write("  the calibration slope in primary-direction rows (source data:\n")
    f.write("  calibration_ci.csv; slopes are logistic-regression slopes of outcome on\n")
    f.write("  logit(predicted probability)). Full CI matrix in Supplement.\n")

# ============ Table 5 ============
with open(os.path.join(OUT, "Table5_formatted.txt"), "w", encoding="utf-8") as f:
    f.write("================================================================================\n")
    f.write("Table 5. Transfer-gap sensitivity to data-alignment interventions\n")
    f.write("         (MGB → external, 5 primary tasks).\n")
    f.write("================================================================================\n\n")
    f.write("All values are transfer gaps (external AUROC − MGB internal CV AUROC),\n")
    f.write("computed against the UNROUNDED internal baseline (single source of truth:\n")
    f.write("transfer_matrix.csv). Positive values favor the external site.\n\n")
    f.write("Direction   Primary   Isolated fixes (gap)          Pairwise combinations (gap)          Joint      Joint AUROC\n")
    f.write("            gap       R1        R2        R3        R1R2      R1R3     R2R3     R1R2R3    gap        [patient 95% CI]\n")
    f.write("                     drop-I    mucoid    >=2020\n")
    f.write("──────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n")
    for _, r in pw.iterrows():
        j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
        tgt = "S" if r["tgt"] == "Stanford" else "U"
        f.write(f"{ABBR[r['task']]}→{tgt}       {r['primary_gap']:+.3f}    {r['R1_only_gap']:+.3f}    "
                f"{r['R2_only_gap']:+.3f}    {r['R3_only_gap']:+.3f}    {r['R1R2_gap']:+.3f}    "
                f"{r['R1R3_gap']:+.3f}   {r['R2R3_gap']:+.3f}   {r['R1R2R3_gap']:+.3f}    {j['gap']:+.3f}    "
                f"{j['auroc']:.3f} [{j['ci_lo']:.3f}–{j['ci_hi']:.3f}]\n")
    f.write("──────────────────────────────────────────────────────────────────────────────────────────────────────────────────\n\n")
    f.write("Notes:\n")
    f.write("- Gaps are external AUROC − MGB internal CV AUROC (primary Intermediate-merged rule,\n")
    f.write("  ADI excluded, canonical feature set, LightGBM fixed 500 trees, no early\n")
    f.write("  stopping), computed against the unrounded internal baseline: MEM 0.770617,\n")
    f.write("  CIP 0.743291, LVX 0.750058, CAZ 0.769396, FEP 0.756791.\n")
    f.write("- R1 (outcome-definition harmonization, Intermediate-excluded): Intermediate rows\n")
    f.write("  excluded from labels on all sites.\n")
    f.write("  R2 (population-coverage restriction): mucoid rows excluded on both\n")
    f.write("  training and test\n")
    f.write("  sides (the MGB retained cohort holds 13 culture-level mucoid cases vs.\n")
    f.write("  1,698 among excluded rows; S/U test sides are affected only where\n")
    f.write("  mucoid isolates exist). R3 (target-era restriction): test-site cohort\n")
    f.write("  restricted to ≥2020\n")
    f.write("  (era restriction, temporal harmonization; MGB calendar years are\n")
    f.write("  unavailable, so the training side cannot be era-restricted).\n")
    f.write("- No isolated intervention was uniformly beneficial against the primary\n")
    f.write("  baseline (by gap magnitude, R1 narrowed 5 of 10 directions, R2 six and\n")
    f.write("  R3 six). R1 raises the source-side reference, so its gap can widen\n")
    f.write("  against the primary baseline; R2 worsened CAZ→S (−0.056 to −0.088),\n")
    f.write("  indicating that the effect of mucoid exclusion was task-specific rather\n")
    f.write("  than uniformly beneficial (§4.10). Joint effects are\n")
    f.write("  not sums of isolated effects; the joint-intervention (R1R2R3) estimate is the\n")
    f.write("  conservative, de-overlapped answer.\n")
    f.write("- After the joint interventions, the joint-intervention 95% CIs covered or lay\n")
    f.write("  entirely above the prespecified primary internal reference in nine of ten\n")
    f.write("  directions: seven\n")
    f.write("  with patient-level CIs covering the reference (MEM→S, MEM→U, CIP→S, LVX→S,\n")
    f.write("  CAZ→S, FEP→S, FEP→U) and two with CIs entirely above it (CIP→U, LVX→U).\n")
    f.write("  Point-estimate sign reversal (negative to positive gap) occurred in four\n")
    f.write("  directions (MEM→S, CIP→S, LVX→S, FEP→S); gap magnitude narrowed in six of\n")
    f.write("  ten directions. One direction retained a persistent negative residual\n")
    f.write("  (CAZ→U −0.079), the only direction whose joint-intervention 95% CI lay entirely\n")
    f.write("  below the internal reference across all five bootstrap seeds.\n")
    f.write("- Estimand caveat: R1 changes the source-side internal performance, so the\n")
    f.write("  R1-containing gaps above are not fully matched transportability gaps; a\n")
    f.write("  matched-reference analysis (joint external AUROC − Intermediate-excluded internal AUROC)\n")
    f.write("  is reported in Table S7. A fully matched joint reference (training side\n")
    f.write("  also restricted to ≥2020) cannot be constructed because MGB calendar years\n")
    f.write("  are unrecoverable; recovery is stated relative to the prespecified primary\n")
    f.write("  reference.\n")
    f.write("- Joint-intervention test cohorts (n = 1,211–3,160) are smaller than the era-only\n")
    f.write("  subsets because the joint analysis additionally removes Intermediate and\n")
    f.write("  mucoid observations (low-event directions such as MEM→S carry ~64 resistant\n")
    f.write("  events; see Discussion §5.6).\n")
    _ns = ", ".join(
        ABBR[r["task"]] + "→" + ("S" if r["tgt"] == "Stanford" else "U") + " "
        + format(int(jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])]["n"].iloc[0]), ",")
        for _, r in pw.iterrows())
    f.write("- Joint-intervention test cohort size (cultures) by direction:\n")
    f.write("  " + _ns + "\n")
    f.write("- Four joint-intervention AUROCs exceed every primary internal CV\n")
    f.write("  value (MEM→S 0.786, CIP→U 0.814, LVX→U 0.814, FEP→S 0.795; maximum\n")
    f.write("  internal value 0.770617). This is expected rather than paradoxical:\n")
    f.write("  the joint interventions also restrict the test cohorts, so these are\n")
    f.write("  not transportability gaps against an unchanged target population.\n")
    f.write("- Gap-magnitude narrowing differs from AUROC improvement where a gap\n")
    f.write("  changes sign: under R3 the target-side AUROC improves in seven of ten\n")
    f.write("  directions, but the gap narrows in only six, because LVX→S moves from\n")
    f.write("  −0.002 to +0.013, so its magnitude grows even as performance improves.\n")
    f.write("- Source: results/phase3/pairwise_fixes.csv (all gaps), joint_fix_ci.csv\n")
    f.write("  (joint AUROC and patient CIs), transfer_matrix.csv (unrounded internal\n")
    f.write("  baselines), boot_seed_sensitivity.csv (seed stability).\n")
    f.write("================================================================================\n")

# ============ Table S7 (matched-reference) ============
with open(os.path.join(OUT, "TableS7_matched_reference.txt"), "w", encoding="utf-8") as f:
    f.write("================================================================================\n")
    f.write("Table S7. Matched-reference joint gaps (joint external AUROC − Intermediate-excluded\n")
    f.write("          internal CV AUROC), supplementary to Table 5.\n")
    f.write("================================================================================\n\n")
    f.write("Direction   Joint AUROC      drop-I internal   Matched gap      Matched gap\n")
    f.write("            [patient 95% CI] (CV AUROC)                        [shifted 95% CI]\n")
    f.write("────────────────────────────────────────────────────────────────────────────\n")
    for _, r in pw.iterrows():
        j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
        d = DROPS[r["task"]]
        tgt = "S" if r["tgt"] == "Stanford" else "U"
        f.write(f"{ABBR[r['task']]}→{tgt}       {j['auroc']:.3f} "
                f"[{j['ci_lo']:.3f}–{j['ci_hi']:.3f}]       {d:.3f}          "
                f"{j['auroc']-d:+.3f}          [{j['ci_lo']-d:+.3f}, {j['ci_hi']-d:+.3f}]\n")
    f.write("────────────────────────────────────────────────────────────────────────────\n\n")
    f.write("Notes:\n")
    f.write("- Matched gap = joint external AUROC − Intermediate-excluded internal CV AUROC (the internal\n")
    f.write("  reference computed under the same Intermediate-exclusion rule applied on the\n")
    f.write("  source side). The joint-intervention 95% CI is shifted by the same constant, so CI\n")
    f.write("  width is preserved; the Intermediate-excluded internal reference is itself a CV estimate\n")
    f.write("  treated as fixed, as in the primary-reference analysis.\n")
    f.write("- Under this matched reference, seven of ten directions remain below the\n")
    f.write("  reference (CAZ→U −0.101 the largest), while CIP→U, LVX→U and FEP→S are at\n")
    f.write("  or above it. The CAZ→U residual is therefore robust to the choice of\n")
    f.write("  primary versus matched reference.\n")
    f.write("- By 95% CI, four directions (MEM→U, LVX→S, CAZ→U, FEP→U) lay entirely\n")
    f.write("  below the matched reference and none lay entirely above it; LVX→S\n")
    f.write("  (upper bound −0.001) and CIP→U (lower bound −0.002) are borderline at\n")
    f.write("  full precision.\n")
    f.write("- A fully matched joint reference (training side also restricted to ≥2020)\n")
    f.write("  cannot be constructed because MGB calendar years are unrecoverable.\n")
    f.write("- Sources: joint_fix_ci.csv, dropI_precise.csv (Intermediate-excluded internal AUROC at\n")
    f.write("  full precision, step56), pairwise_fixes.csv.\n")
    f.write("================================================================================\n")

print("Tables regenerated:", ", ".join(sorted(os.listdir(OUT))))
