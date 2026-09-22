# -*- coding: utf-8 -*-
"""
ARMD 阶段3 — 跨系统迁移审计 (6 方向 × 5 主任务)
- 训练侧全量训练 LightGBM, 目标侧独立测试
- 指标: AUROC/AUPRC + 校准(Brier/截距/斜率) + 迁移缺口(外部-内部)
- 特征证据漂移: 同一模型在三站数据上的 SHAP, 稳定/捷径清单对照
输出: E:\\ARMD\\results\\phase3\\
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
os.makedirs(OUT, exist_ok=True)
PRIMARY = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SITES = ["MGB","Stanford","UTSW"]
DROP_COLS = ["order_proc_id_coded","anon_id","label","fold_id","prior_pa_res","time","ward_raw"]
# 与 step14 主模型保持一致 (adi 已从主模型剔除, 2026-08-07 评审驱动)
EXCLUDE_FEATURES = ["adi"]

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep
import lightgbm as lgb
import shap
from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import spearmanr

# 固定类别映射 (在 MGB 上拟合, 应用到 S/U) — 防跨站编码错位
def fit_cat_maps(mgb_df):
    maps = {}
    for c in ["age_bin","ward_h","specimen"]:
        vals = sorted(mgb_df[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps

def apply_cat(df, maps):
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    return df

def prep_cross(df, maps, adi_median=None):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years","",regex=False)
    df = apply_cat(df, maps)
    if adi_median is not None:
        df["adi"] = df["adi"].fillna(adi_median)   # 标准化参数只来自训练侧
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

def feats_of(df):
    return [c for c in df.columns if c not in DROP_COLS and c != "ward_raw" and c not in EXCLUDE_FEATURES]

def calib_metrics(y, p):
    from sklearn.linear_model import LogisticRegression
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(max_iter=500)
    lr.fit(lp.reshape(-1, 1), y)
    return brier, float(lr.intercept_[0]), float(lr.coef_[0][0])

report = open(os.path.join(OUT, "report_phase3.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); report.write(line + "\n")

if __name__ != "__main__":
    raise RuntimeError("step16 为独立运行脚本, 禁止 import (防止连带执行阶段3)")
internal = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "cv_internal.csv"))
internal = internal[internal["model"] == "lgb"].groupby("task")["auroc"].mean().to_dict()
stable = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "stable_features.csv"))
stable_set = set(stable[(stable["rank_stability"] >= 0.7) & (stable["sign_consistency"] >= 0.75)]["feature"])

rows = []
drift_rows = []
for task in PRIMARY:
    p(f"\n{'='*90}\n===== {task} =====\n{'='*90}")
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw)
    adi_med = mgb_raw["adi"].median()               # 训练侧统计量
    mgb = prep_cross(mgb_raw, maps, adi_med)
    feats = feats_of(mgb)

    for src in SITES:
        for tgt in SITES:
            if src == tgt: continue
            t0 = time.time()
            tr = prep_cross(pd.read_csv(os.path.join(CLEAN, src, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, adi_med)
            te = prep_cross(pd.read_csv(os.path.join(CLEAN, tgt, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"), maps, adi_med)
            m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                                   min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                                   random_state=42, verbose=-1)
            m.fit(tr[feats], tr["label"])
            pte = m.predict_proba(te[feats])[:, 1]
            au = roc_auc_score(te["label"], pte)
            ap = average_precision_score(te["label"], pte)
            brier, ci, cs = calib_metrics(te["label"], pte)
            gap = au - internal.get(task, np.nan)
            rows.append(dict(task=task, src=src, tgt=tgt, n_test=len(te), auroc=au, auprc=ap,
                             brier=brier, calib_int=ci, calib_slope=cs,
                             internal_auroc=internal.get(task), transfer_gap=gap))
            p(f"[{src}→{tgt}] AUROC {au:.3f} (缺口 {gap:+.3f}) | AUPRC {ap:.3f} | Brier {brier:.3f} | 校准截距 {ci:+.2f} 斜率 {cs:.2f} | {time.time()-t0:.0f}s")
            # 特征证据漂移 (MGB→S/U 主方向)
            if src == "MGB" and tgt in ("Stanford","UTSW"):
                ex = shap.TreeExplainer(m)
                sv_tr = ex.shap_values(tr[feats])
                sv_tr = sv_tr[1] if isinstance(sv_tr, list) else sv_tr
                sv_te = ex.shap_values(te[feats])
                sv_te = sv_te[1] if isinstance(sv_te, list) else sv_te
                for i, f in enumerate(feats):
                    imp_tr = np.abs(sv_tr[:, i]).mean(); imp_te = np.abs(sv_te[:, i]).mean()
                    drift_rows.append(dict(task=task, tgt=tgt, feature=f, shap_mgb=imp_tr, shap_tgt=imp_te,
                                           ratio=imp_te / max(imp_tr, 1e-9),
                                           in_stable_set=int(f in stable_set)))

pd.DataFrame(rows).to_csv(os.path.join(OUT, "transfer_matrix.csv"), index=False)
pd.DataFrame(drift_rows).to_csv(os.path.join(OUT, "feature_drift.csv"), index=False)

# 漂移摘要: 稳定特征 vs 非稳定特征
p("\n" + "=" * 90)
p("特征证据漂移摘要 (MGB→S/U, 按特征组)")
drift = pd.DataFrame(drift_rows)
for tgt in ["Stanford","UTSW"]:
    d = drift[drift["tgt"] == tgt]
    st = d[d["in_stable_set"] == 1]; ns = d[d["in_stable_set"] == 0]
    p(f"[{tgt}] 稳定特征平均 SHAP 比值: {st['ratio'].mean():.2f} | 非稳定: {ns['ratio'].mean():.2f}")
    worst = st.sort_values("ratio").head(8)
    p(f"  稳定特征中最漂移: " + " > ".join(f"{r.feature}({r.ratio:.2f})" for r in worst.itertuples()))
p("\n汇总已存 results/phase3/ (transfer_matrix.csv, feature_drift.csv)")
report.close()
print("完成", flush=True)
