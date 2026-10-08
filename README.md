# ARMD Cross-System Transferability Audit — Code Release

[![DOI](https://zenodo.org/badge/1380814326.svg)](https://doi.org/10.5281/zenodo.22888421)

Analysis code for a cross-system transferability audit of a *Pseudomonas aeruginosa*
antimicrobial-susceptibility prediction model, trained at Mass General Brigham (MGB)
and externally evaluated at Stanford Health Care and UT Southwestern (UTSW).

---

## Part 1 — Where the data is

Three public, de-identified EHR datasets. **Raw and cleaned data are not
redistributed with this code**, per the data-use agreements.

| Dataset | Repository | Access |
|---|---|---|
| **ARMD-MGB** | PhysioNet · `10.13026/2r5k-b955` | **Credentialed.** Requires PhysioNet credentialed-user status, CITI "Data or Specimens Only Research" training, and a signed DUA. |
| **ARMD-Stanford** | Dryad · `10.5061/dryad.jq2bvq8kp` | CC0 — open |
| **ARMD-UTSW** | Dryad · `10.5061/dryad.0rxwdbsd5` | CC0 — open |

`frozen_dataset_hashes.csv` lists the SHA-256 of every file in the frozen cleaned
dataset (v3.3, 31 files) used for the reported analysis. Use it to confirm your
cleaned data matches what produced the published numbers.

---

## Part 2 — How to run it

**Environment.** Python 3.13.2 (Windows 11); dependencies pinned in
`requirements.txt`.

```bash
pip install -r requirements.txt
```

### Step 1 — Point the scripts at your data

The scripts are published **unmodified** and still contain the original analysis
machine's absolute paths — `E:\ARMD\` (raw data, cleaned data, results tree) and
`F:\E\Machine Learning\ARMD\` (the project tree). Neither exists on your machine.

Two things need substituting:

- **30 scripts** join `BASE + "data" + "clean"` — the cleaned data now lives in
  `clean/clean/`.
- **17 scripts** reference `E:\ARMD\` for the raw datasets.

The paths are module-level constants (`BASE`, `ROOT`, `OUT`, `CLEAN`, `R3`) near
the top of each file, so substitution is mechanical. Per-script detail is in
`SCRIPTS.md`.

We have deliberately **not** refactored this into a config module: the published
paths are the ones that produced the reported numbers, and a refactor we could
not re-verify end to end would trade a known-good pipeline for an untested one.

### Step 2 — Run the chain

```text
step9_clean.py                  # cleaning (point-in-time discipline)
step14_model_phase2.py          # MGB internal CV, care-setting gradient, SHAP, ablation
step16_phase3_transfer.py       # cross-system transfer + feature drift
step21_shift_decomposition.py   # label / covariate / concept shift
step28 – step59                 # sensitivity and verification chain
figures_v5.py                   # Figures 1–3, S1–S4
figures_v5_ledger.py            # Figure 4
```

### Step 3 — Verify

`step52_verify_chain.py` is the acceptance test. Run it after any rebuild; it
must end `RESULT: ALL PASS`.

```bash
python step52_verify_chain.py
```

### All 55 scripts in this release

```text
figures_v5                figures_v5_ledger         spike1_mgb_mem_pipeline
step2_main_tables_probe   step3_stanford_pa_audit   step4_jitter_sdd_dtr_audit
step5_utsw_probe          step6_crosswalk_probe     step7_contract_check
step8_dict_freeze         step9_clean               step10_reconcile
step11_spotcheck          step12_measure            step13_debug_f7
step14_model_phase2       step15_stable_features    step16_phase3_transfer
step17_spotcheck_drift    step18_verify             step19_check_f6
step20_check_proc_su      step21_shift_decomposition step22_stable_subset_dca
step23_covariate_fix      step24_read_dca           step25_adi_final
step26_review_check       step27_check_review       step28_era_sensitivity
step29_cov_no_adi         step30_adi_sensitivity    step31_ipw_era_ci
step32_label_coverage     step33_verify_mucoid      step34_check_specimen
step35_read_new_drift     step36_verify_numbers     step37_verify_codes
step38_mucoid_sensitivity step39_tables             step40_table3_table4
step46_I_sensitivity      step47_independent_recalc step48_patient_bootstrap_all
step49_attribution_ledger step50_joint_fix_ci       step51_pairwise_fixes
step52_verify_chain       step53_boot_seed_sensitivity step54_imputation_sensitivity
step55_calibration_ci     step57_regenerate_tables  step58_qa_audit
step59_joint_ablation
```

`step1`, `step45` and `step56` do not exist. What each script does — and the
per-script path audit — is in `SCRIPTS.md`.

---

## What will not run as-is

- **Anything rebuilding cohorts from the raw tables.** The raw data is
  credential-gated and not redistributed, so `step9_clean.py` and the steps
  downstream of it cannot be executed from this release alone.
- **`step56`.** It produced `dropI_precise.csv`, which `step57` and `step58`
  read. That script is not included, and the CSV is not redistributed, so this
  intermediate cannot be regenerated.
- **`step58_qa_audit.py`** additionally reads manuscript-side files that live
  outside this release (`build_docx.py`, the formatted tables, the manuscript
  directory) and requires `python-docx`.

Everything downstream of the frozen result CSVs does run; this was verified
against the published numbers.

---

## License

MIT — see `LICENSE`.

## Citation

See `CITATION.cff`. Please cite the paper (citation to be completed on
publication) and the three datasets listed above.

Archived on Zenodo:

- Version **v1.0.0**: `10.5281/zenodo.22888422`
- All versions: `10.5281/zenodo.22888421`
