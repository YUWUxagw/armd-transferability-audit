# -*- coding: utf-8 -*-
"""
Step 48 — TRUE patient-level bootstrap for all 30 transfer directions.
The earlier stable_subset.csv boot_ci resampled rows (not patients); this is a
methodological correction. Patient-level bootstrap resamples patients and
keeps all their cultures together. 200 iterations per direction.
"""
import os, sys
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_patient_bootstrap.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols
from sklearn.metrics import roc_auc_score

def prep_cross_fillna(df, maps, adi_med):
    """Same encoding semantics as the main pipeline (step16 local prep_cross):
    unmapped target categories are encoded as -1 (source-out-of-domain)."""
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years", "", regex=False)
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    df["adi"] = df["adi"].fillna(adi_med)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df
prep_cross = prep_cross_fillna

TASKS  = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SRCS   = ["MGB","Stanford","UTSW"]
TGTS   = ["MGB","Stanford","UTSW"]
NB     = 200
src_cache = {}

def patient_boot(y, p, pid, n_boot=NB, seed=0):
    """Resample PATIENTS (keep all cultures of a patient together)."""
    rng = np.random.RandomState(seed)
    pids = np.unique(pid)
    aucs = []
    for _ in range(n_boot):
        sel = rng.choice(pids, size=len(pids), replace=True)
        # build index mask: cultures whose patient is in sel (with replacement
        # handled by weighting: sample with replacement at patient level and
        # expand counts)
        cnt = pd.Series(sel).value_counts()
        idx = []
        for pid_i, c in cnt.items():
            idx.extend(np.where(pid == pid_i)[0].tolist() * c)
        idx = np.array(idx)
        if len(idx) < 2 or len(np.unique(y[idx])) < 2:
            continue
        try:
            aucs.append(roc_auc_score(y[idx], p[idx]))
        except ValueError:
            continue
    if len(aucs) < 50:
        return None, None
    return float(np.percentile(aucs, 2.5)), float(np.percentile(aucs, 97.5))

rows = []
for t in TASKS:
    feats_all = None
    models = {}
    te_cache = {}
    for src in SRCS:
        raw = pd.read_csv(os.path.join(CLEAN, src, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        maps = fit_cat_maps(raw)
        adi = raw["adi"].median()
        mgb_prep = prep_cross(raw, maps, adi)
        feats = feature_cols(mgb_prep)
        feats_all = feats
        models[src] = train_lgb(mgb_prep[feats], mgb_prep["label"], mgb_prep[feats], mgb_prep["label"])
        src_cache[src] = (maps, adi)
    for tgt in TGTS:
        raw_t = pd.read_csv(os.path.join(CLEAN, tgt, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        for src in SRCS:
            if src == tgt:
                continue
            # encode target with the SOURCE's fitted maps (consistency with training)
            maps_src, adi_src = src_cache[src]
            te = prep_cross(raw_t, maps_src, adi_src)
            X = te[[f for f in feats_all if f in te.columns]]
            y = te["label"].values
            pred = models[src].predict_proba(X)[:, 1]
            lo, hi = patient_boot(y, pred, te["anon_id"].values)
            auc = roc_auc_score(y, pred)
            rows.append(dict(task=t, src=src, tgt=tgt, auroc=round(auc, 3),
                             ci_lo=round(lo, 3) if lo else None, ci_hi=round(hi, 3) if hi else None))
            p(f"  {t} {src}→{tgt}: AUROC {auc:.3f} patient-CI [{lo:.3f}-{hi:.3f}]")

pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_ci_patient.csv"), index=False)
p("\nsaved: transfer_ci_patient.csv")
rep.close()
print("Done", flush=True)
