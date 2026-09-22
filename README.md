# ARMD Cross-System Transferability Audit — Code Release

[![DOI](https://zenodo.org/badge/1380814326.svg)](https://doi.org/10.5281/zenodo.22888421)

Analysis code for a cross-system transferability audit of a *Pseudomonas aeruginosa*
antimicrobial-susceptibility prediction model trained at one US health system
(Mass General Brigham) and evaluated at two others (Stanford Health Care,
UT Southwestern Medical Center).

Pipeline architecture: cleaning (step9) → phase-2 modeling (step14) →
phase-3 cross-system transfer (step16) → shift decomposition (step21) →
sensitivity and verification chain (steps 28–59) → paper figures
(figures_v4, figures_v5_ledger).

## Data

Three public, de-identified EHR datasets (see paper Methods §1):

- **ARMD-MGB**: PhysioNet, DOI 10.13026/2r5k-b955 — *credentialed access*
  (PhysioNet Credentialed Health Data License 1.5.0; requires credentialed-user
  status, CITI "Data or Specimens Only Research" training, and a signed DUA)
- **ARMD-Stanford**: Dryad, DOI 10.5061/dryad.jq2bvq8kp (CC0)
- **ARMD-UTSW**: Dryad, DOI 10.5061/dryad.0rxwdbsd5 (CC0)

**Raw and cleaned data are not redistributed with this code**, per the data-use
agreements. To reproduce the analysis, obtain the three datasets from the
repositories above.

`frozen_dataset_hashes.csv` lists the SHA-256 of every file in the frozen
cleaned dataset (v3.3, 31 files) used for the reported analysis. The per-task
row counts and a 12-character SHA-256 prefix for each task file are additionally
recorded in `manifest.csv`, which the cleaning step writes.

## Environment

Python 3.13.2 (Windows 11); dependencies pinned in `requirements.txt`.
Verified on the frozen dataset v3.3, 2026-08-10.

## Run order

```
step9_clean.py            # site-parameterized cleaning (point-in-time discipline)
step14_model_phase2.py    # MGB internal CV, care-setting gradient, SHAP, ablation
step16_phase3_transfer.py # cross-system transfer + feature drift
step21_shift_decomposition.py  # label/covariate/concept shift under explicit assumptions
step28_era_sensitivity.py # breakpoint-era sensitivity (>=2020)
step38_mucoid_sensitivity.py  # mucoid subgroup sensitivity
step46_I_sensitivity.py   # Intermediate-exclusion (drop-I) sensitivity
step47_independent_recalc.py  # cohort reproducibility + sklearn-only metrics + LR family
step48_patient_bootstrap_all.py  # patient-level bootstrap CIs, all 30 directions
step49_attribution_ledger.py    # fixable-gap attribution ledger
step50_joint_fix_ci.py          # joint-fix patient-level CIs (10 directions)
step51_pairwise_fixes.py        # pairwise combination cross-validation (8 combos)
step52_verify_chain.py          # FULL-CHAIN REGRESSION VERIFICATION (run after any
                                # rebuild; A1-A7 asserts; must end ALL PASS)
step53_boot_seed_sensitivity.py # bootstrap seed stability (5 seeds x 200)
step54_imputation_sensitivity.py# missing-data imputation rule sensitivity
step58_qa_audit.py              # manuscript-level QA (numbers vs CSVs, banned phrases)
figures_v4.py / figures_v5_ledger.py  # publication figures (SVG+PDF+TIFF)
```

`step2`–`step8` and `step10`–`step13` are data-provenance, crosswalk and
vocabulary-freezing steps that document how the three sites were harmonized and
how drug-name dictionaries were fixed; they are included for completeness.

## ⚠️ Path configuration (read before running)

**This release preserves the analysis environment's original absolute paths.**
The scripts reference two roots that existed on the analysis machine:

- `E:\ARMD\` — raw datasets, cleaned data (`data\clean\`), intermediate audit
  reports (`audit_out\`), and the original results tree (`results\phase2|phase3\`)
- `F:\E\Machine Learning\ARMD\` — the project tree as it stood at submission
  (`05_源数据\` for results, `clean\clean\` for cleaned data, `02_图表\` for figures)

Scripts from different pipeline generations point at different roots.

**A path audit of this release (2026-09-22) found that 31 of the 62 scripts
build paths that do not exist in the project tree as it stands**, in two
patterns:

1. **`data/clean` (~20 scripts)** — the cleaning output was later moved to
   `clean/clean/`, but the scripts still join `BASE + data + clean`. Affects
   `step14`–`step16`, `step21`–`step23`, `step26`, `step28`–`step31`,
   `step36`, `step38`, `step39`, `step48`, `step52`–`step55` and others.
2. **Raw datasets joined onto the project root (10 scripts)** — `step9`,
   `step11`, `step18`, `step27`, `step32`, `step46`, `step47`, `step49`,
   `step50`, `step51` build `BASE/ARMD-MGB` etc., but the raw datasets live
   under `E:\ARMD\`, not inside the project directory.

The reporting and QA scripts that were kept in active use after the project
moved (notably `step58_qa_audit.py`, which reads only `05_源数据/` and the
manuscript directory) do resolve correctly.

Before running, either recreate the expected layout or substitute your own
paths. The roots are module-level constants (`BASE`, `ROOT`, `OUT`, `CLEAN`)
near the top of each file, so substitution is mechanical.

**We have deliberately not refactored this to a single config module.** The
published paths are the ones that produced the reported numbers; a refactor we
could not re-verify end to end would trade a known-good pipeline for an
untested one. If you need a parameterized version and can run the full chain,
`step52_verify_chain.py` is the acceptance test — it must end `ALL PASS`.

## Verification discipline

- Any analysis that rebuilds cohorts from raw tables must first pass
  `step52_verify_chain.py` (RESULT: ALL PASS). The chain asserts: A1 feature-set
  identity with canonical task files; A2 primary reproduction of the canonical
  transfer matrix (n and AUROC); A3–A5 cross-script consistency of the
  ledger/joint-fix/pairwise outputs; A6 CI legality; A7 completeness.
- Random seeds are fixed and logged by the steps that use them;
  `step53_boot_seed_sensitivity.py` re-runs the bootstrap under five seeds, and
  the seed-stability of the reported residual is a reported result.

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. If you use this code, please cite the paper (citation to be
completed on publication) and the three datasets listed above.

This release is archived on Zenodo:

- Version **v1.0.0**: `10.5281/zenodo.22888422`
- All versions: `10.5281/zenodo.22888421`
