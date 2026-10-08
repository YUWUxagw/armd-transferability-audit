# Script index and path audit

Supplementary detail for the code release. The main `README.md` covers only
where the data is and how to run the chain; this file covers what each script
does and where the published paths currently point.

---

## 1. Run order (main chain)

```text
step9_clean.py                 # site-parameterized cleaning (point-in-time discipline)
step14_model_phase2.py         # MGB internal CV, care-setting gradient, SHAP, ablation
step16_phase3_transfer.py      # cross-system transfer + feature drift
step21_shift_decomposition.py  # label/covariate/concept shift under explicit assumptions
step28_era_sensitivity.py      # breakpoint-era sensitivity (>=2020)
step38_mucoid_sensitivity.py   # mucoid subgroup sensitivity
step46_I_sensitivity.py        # Intermediate-exclusion (drop-I) sensitivity
step47_independent_recalc.py   # cohort reproducibility + sklearn-only metrics + LR family
step48_patient_bootstrap_all.py# patient-level bootstrap CIs, all 30 directions
step49_attribution_ledger.py   # fixable-gap attribution ledger
step50_joint_fix_ci.py         # joint-fix patient-level CIs (10 directions)
step51_pairwise_fixes.py       # pairwise combination cross-validation (8 combos)
step52_verify_chain.py         # FULL-CHAIN REGRESSION VERIFICATION — see §4
step53_boot_seed_sensitivity.py# bootstrap seed stability (5 seeds x 200)
step54_imputation_sensitivity.py  # missing-data imputation rule sensitivity
step58_qa_audit.py             # manuscript-level QA (numbers vs CSVs, banned phrases)
figures_v5.py                  # Figures 1-3 and S1-S4 (SVG + PDF + TIFF)
figures_v5_ledger.py           # Figure 4
```

`step2`–`step8` and `step10`–`step13` are data-provenance, crosswalk and
vocabulary-freezing steps that document how the three sites were harmonized and
how drug-name dictionaries were fixed; they are included for completeness.

Not listed above but **required as producers**, because later steps read their
outputs: `step15_stable_features.py` (writes `phase2/stable_features.csv`),
`step22_stable_subset_dca.py` (writes `dca.csv`, `stable_subset.csv`),
`step39_tables.py` / `step40_table3_table4.py` (tables),
`step57_regenerate_tables.py` (writes the `Table*_formatted.txt` files that
`step58` opens), `step59_joint_ablation.py` (writes `joint_ablation.csv`) and
`step60_model_family_external.py` (writes `model_family_external.csv` and
`model_family_external_calibration.csv`).

`step60_model_family_external.py` was added on 2026-10-08 to close a
reproducibility gap: those two CSVs had been in the results tree since
2026-09-21 and are cited in the manuscript, but no shipped script produced them.
It trains logistic regression on the source cohort and evaluates on the target
cohort (no cross-validation), takes the LightGBM column from the frozen
`transfer_matrix.csv` rather than refitting, and verifies both output files
against the published versions before writing; the external table reproduces
byte-for-byte. It deliberately does not import `step16_phase3_transfer.py`, which
pulls in `shap` at module level, so the one calibration function it needs is
inlined instead.

The script shipped without any README coverage is `spike1_mgb_mem_pipeline.py`.

Step numbers are **not contiguous**: `step1`, `step45` and `step56` do not exist.
`step56` is the producer of `dropI_precise.csv`, which `step57` and `step58`
read — that intermediate cannot be regenerated from this release.

---

## 2. Figure scripts

`figures_v5.py` and `figures_v5_ledger.py` are the live producers.

Earlier revisions of this release also carried `figures_v2.py`, `figures_v3.py`,
`figures_v4.py`, `step41_fig1_flowchart.py`, `step42_fig2_heatmap.py`,
`step43_fig3_calibration.py` and `step44_supp_figs.py`. These were **superseded
generations** and have been removed, because they emit filenames from before the
2026-09-21 figure renumbering — `step42` writes `Figure2_feature_drift_heatmap`
while the published figure is `Figure3_*`, and `step44`'s S1/S2/S3 are swapped —
so running them silently overwrites correct figures with mislabelled ones.

**Reproducibility caveat.** `figures_v5_ledger.py` regenerates Figure 4 from the
frozen CSVs. `figures_v5.py` regenerates three of the seven figures
byte-for-byte (`Figure1`, `FigureS1`, `FigureS2`); the remaining four
(`Figure2`, `Figure3`, `FigureS3`, `FigureS4`) differ in rendered byte size when
regenerated, which is consistent with a different matplotlib/font environment.
The encoded values are unchanged. Treat figure regeneration as
visually-equivalent rather than byte-identical.

---

## 3. Path audit

The scripts are published unmodified, so they still contain the analysis
machine's absolute paths. Two roots are referenced:

- `E:\ARMD\` — raw datasets, cleaned data (`data\clean\`), intermediate audit
  reports (`audit_out\`), and the original results tree (`results\phase2|phase3\`)
- `F:\E\Machine Learning\ARMD\` — the project tree as it stood at submission
  (`05_源数据\` for results, `clean\clean\` for cleaned data, `02_图表\` for figures)

Scripts from different pipeline generations point at different roots. Two
patterns remain:

**Pattern 1 — `data/clean` (24 scripts).** The cleaning output was later moved to
`clean/clean/`, but these scripts still join `BASE + "data" + "clean"`. Affects
`step9`, `step10`, `step14`, `step15`, `step16`, `step22`, `step23`,
`step26`–`step31`, `step36`, `step38`, `step39`, `step48`–`step51`,
`step53`–`step55` and `step59`.

**Pattern 2 — `E:\ARMD\` (17 scripts).** These reference the original analysis
root in their path constants. Affects `spike1_mgb_mem_pipeline.py`, `step2`–`step8`,
`step10`, `step12`, `step13`, `step19`, `step33`, `step34`, `step39`,
`figures_v5.py` and `step58` (in a comment).

Scripts whose paths were updated on 2026-10-08 and now resolve correctly:
`step11`, `step17`, `step18`, `step20`, `step21`, `step24`, `step25`,
`step35`, `step37`, `step46`, `step47`, `step52` and
`step60_model_family_external.py`. `step57_regenerate_tables.py` and
`figures_v5_ledger.py` already resolved.

`step47_independent_recalc.py` was also corrected on 2026-10-08: its five-fold
cross-validation now uses the pipeline's patient-level `fold_id` rather than
`StratifiedKFold` on row indices. The previous comment described the split as
patient-level while the code split rows, which put roughly 19% of patients on
both sides of a fold (900 of 4,744 in the meropenem cohort) and inflated the
cross-validated scores, much more for the gradient-boosted arm (+0.013 to
+0.030) than for logistic regression (+0.005 to +0.011). With the split aligned
to the main analysis, the script's logistic-regression arm reproduces the
pipeline's `cv_internal.csv` to within 0.0004, so remaining differences are
attributable to implementation rather than to splitting. Section (1) now skips
with a message when the credential-gated MGB raw table is absent, so sections
(2) and (3) run on any machine that has the frozen task files.

`step58_qa_audit.py` does **not** resolve despite being the manuscript-level QA
script: besides `05_源数据/` it also reads `04_代码/build_docx.py`,
`04_代码/figures_v5_ledger.py`, the `Table*_formatted.txt` files and the
manuscript directory, none of which ship with this release. It also imports
`python-docx`.

Before running, either recreate the expected layout or substitute your own
paths. The roots are module-level constants (`BASE`, `ROOT`, `OUT`, `CLEAN`)
near the top of each file, so substitution is mechanical.

---

## 4. Verification discipline

- Any analysis that rebuilds cohorts from raw tables must first pass
  `step52_verify_chain.py` (`RESULT: ALL PASS`). The chain asserts: A1 feature-set
  identity with canonical task files; A2 primary reproduction of the canonical
  transfer matrix (n and AUROC); A3–A5 cross-script consistency of the
  ledger/joint-fix/pairwise outputs; A6 CI legality; A7 completeness.
- Random seeds are fixed and logged by the steps that use them;
  `step53_boot_seed_sensitivity.py` re-runs the bootstrap under five seeds, and
  the seed-stability of the reported residual is a reported result.
