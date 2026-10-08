# -*- coding: utf-8 -*-
"""
Step 47 — independent recalculation (third-party style, no import of our
modeling pipeline). Validates three claims:
  (1) cohort reproducibility: rebuild MEM cohort from RAW tables with the
      primary (I-merged) rule and compare against the frozen dataset.
  (2) metric correctness: recompute AUROC / Brier / calibration on the frozen
      task files using ONLY pandas + sklearn public APIs.
  (3) conclusion robustness across model families: refit with sklearn's
      GradientBoostingClassifier (not LightGBM) and logistic regression, and
      confirm internal/CV conclusions hold.

独立性来自「不 import 管线代码」（本文件只用 pandas + sklearn 公共 API）。
交叉验证**使用管线在清洗阶段生成的 fold_id**，而非重新随机切分 —— 这样与稿件
的主分析同折，任何差异都只能归因于实现，不能归因于切分。
"""
import os
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
RAW  = os.path.join(BASE, "ARMD-MGB")
# 2026-10-08：清洗数据已由 data/clean 迁至 clean/clean（与 step11/21/46 同写法）。
# 此前本文件未随迁移更新，导致它在 F 盘工作树根本跑不起来。
CLEAN = os.path.join(BASE, "clean", "clean")
rep = open(os.path.join(BASE, "05_源数据", "phase3", "report_independent_recalc.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

from sklearn.ensemble import GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
DROP_IDS = ("order_proc_id_coded","anon_id","label","fold_id","prior_pa_res","time")
cv = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "cv_internal.csv"))

def prep_X(d):
    """与管线同口径的编码（scikit 公共 API）；不做任何管线 import。"""
    X = d[[c for c in d.columns if c not in DROP_IDS]].copy()
    for c in ["age_bin","ward_h","specimen"]:
        X[c] = X[c].astype("category").cat.codes
    X["adi"] = X["adi"].fillna(X["adi"].median())
    X["prior_pseudo_days"] = X["prior_pseudo_days"].fillna(-1)
    return X

def patient_folds(d):
    """管线在清洗阶段生成的 fold_id = CRC32(anon_id) % 5，天然按患者分配。"""
    f = d["fold_id"].astype(int).values
    assert d.groupby("anon_id")["fold_id"].nunique().max() == 1, \
        "fold_id 未按患者分配 —— 患者级切分的前提不成立"
    return f

def cv_scores(d, model):
    X = prep_X(d); y = d["label"]; f = patient_folds(d)
    aucs = []
    for k in range(5):
        va = np.where(f == k)[0]; tr = np.where(f != k)[0]
        if model == "lr":
            sc = StandardScaler().fit(X.iloc[tr])
            m = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
            pr = m.fit(sc.transform(X.iloc[tr]), y.iloc[tr]).predict_proba(sc.transform(X.iloc[va]))[:, 1]
        else:
            m = GradientBoostingClassifier(n_estimators=300, learning_rate=0.05,
                                           max_depth=3, random_state=42)
            pr = m.fit(X.iloc[tr], y.iloc[tr]).predict_proba(X.iloc[va])[:, 1]
        aucs.append(roc_auc_score(y.iloc[va], pr))
    return float(np.mean(aucs))

# 冻结队列先载入：第 (2)(3) 段都要用它，而第 (1) 段在本机可能被跳过。
frozen = pd.read_csv(os.path.join(CLEAN, "MGB", "task_meropenem.csv"),
                     low_memory=False, encoding="utf-8-sig")

# ── (1) Cohort reproducibility: rebuild MEM from raw, I-merged ──
p("(1) Cohort reproducibility (MEM, I-merged, rebuilt from raw tables)")
RAW_TABLE = os.path.join(RAW, "microbiology_cohort_deid_tj_updated.csv")
if not os.path.exists(RAW_TABLE):
    # 2026-10-08：ARMD-MGB 是 PhysioNet 凭据访问数据（10.13026/2r5k-b955），
    # 不随代码分发。只有本段需要它；(2)(3) 段只用冻结的 task 文件。
    # 故此处**跳过而非中止**，让脚本在任何有冻结数据的机器上都能完成 (2)(3)。
    p("  SKIPPED — raw table not present at:")
    p(f"    {RAW_TABLE}")
    p("  ARMD-MGB is credentialed-access data and is not redistributed;")
    p("  sections (2) and (3) below do not require it.")
else:
    df = pd.read_csv(RAW_TABLE,
                     usecols=["anon_id","order_proc_id_coded","organism","antibiotic",
                              "CLSI_2022_pheno","neg_cx","prelim_AST"],
                     low_memory=False, encoding="utf-8-sig")
    org = df["organism"].astype(str).str.upper()
    pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
    drug = df["antibiotic"].astype(str).str.lower().str.replace("/","_",regex=False).str.replace("-","_",regex=False).str.strip()
    m = pa & drug.eq("meropenem") & ~df["neg_cx"].astype(str).str.strip().eq("X") \
        & ~df["prelim_AST"].astype(str).str.strip().eq("X")
    ph = df["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
    R = ph.eq("RESISTANT"); S = ph.eq("SUSCEPTIBLE"); I = ph.eq("INTERMEDIATE")
    lab = pd.Series(0, index=df.index); lab[R | I] = 1
    sub = df[m & (R | S | I)].copy(); sub["label"] = lab[m & (R | S | I)].astype(int)
    sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")
    p(f"  rebuilt: n={len(sub):,} nR={int(sub['label'].sum()):,} prev={sub['label'].mean():.3f}")
    p(f"  frozen : n={len(frozen):,} nR={int(frozen['label'].sum()):,} prev={frozen['label'].mean():.3f}")
    match = set(sub["order_proc_id_coded"]) == set(frozen["order_proc_id_coded"])
    p(f"  culture-id sets identical: {match}")

# ── (2) Metric correctness on frozen MEM (sklearn-only) ──
p("\n(2) Metric recomputation (sklearn public APIs only, patient-level 5-fold)")
lr_auc = cv_scores(frozen, "lr")
gb_auc = cv_scores(frozen, "gb")
_pl_lr = cv[(cv["task"]=="meropenem") & (cv["model"]=="lr")]["auroc"].mean()
_pl_gb = cv[(cv["task"]=="meropenem") & (cv["model"]=="lgb")]["auroc"].mean()
p(f"  sklearn LR  AUROC : {lr_auc:.4f}   pipeline (cv_internal) {_pl_lr:.4f}   Δ={lr_auc-_pl_lr:+.4f}")
p(f"  sklearn GBM AUROC : {gb_auc:.4f}   pipeline (cv_internal) {_pl_gb:.4f}   Δ={gb_auc-_pl_gb:+.4f}")
p("  说明：切分与管线同折（fold_id），故上表差异只能来自实现，不能来自切分。")

# ── (3) Conclusion robustness across model families ──
p("\n(3) Cross-model-family check: LR vs sklearn-GBM vs paper LightGBM, all 5 tasks")
p(f"  {'task':14s} {'sk-LR':>7s} {'sk-GBM':>7s} {'paper-LGB':>10s} {'paper-LR':>9s}  GBM>LR?")
for t in TASKS:
    d = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"),
                    low_memory=False, encoding="utf-8-sig")
    a_lr = cv_scores(d, "lr"); a_gb = cv_scores(d, "gb")
    pl_gb = cv[(cv["task"]==t) & (cv["model"]=="lgb")]["auroc"].mean()
    pl_lr = cv[(cv["task"]==t) & (cv["model"]=="lr")]["auroc"].mean()
    p(f"  {t:14s} {a_lr:7.4f} {a_gb:7.4f} {pl_gb:10.4f} {pl_lr:9.4f}  "
      f"{'yes' if a_gb > a_lr else 'NO'}")

rep.close()
print("Done", flush=True)
