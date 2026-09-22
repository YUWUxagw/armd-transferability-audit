# -*- coding: utf-8 -*-
"""
ARMD 阶段2 收尾 — 候选稳定特征集 (MGB 内部场景稳定性)
方法: 每任务训练 4 个场景模型(留一场景的互补集), 全量 SHAP
  稳定性 = ①重要性排序跨场景 Spearman 相关  ②SHAP 方向(符号)一致性
  候选稳定 = 高平均重要性 + 高排序稳定 + 方向一致
输出: E:\\ARMD\\results\\phase2\\stable_features.csv + report
"""
import os
import numpy as np
import pandas as pd
from scipy.stats import spearmanr

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean", "MGB")
OUT = os.path.join(BASE, "05_源数据", "phase2")
PRIMARY = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SUSPECTS = ["mucoid","ward_h","specimen","icu","adi"]
DROP_COLS = ["order_proc_id_coded","anon_id","label","fold_id","prior_pa_res","time","ward_raw"]

import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import prep, feature_cols
import lightgbm as lgb
import shap

report = open(os.path.join(OUT, "report_stable_features.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); report.write(line + "\n")

all_rows = []
for task in PRIMARY:
    df = prep(pd.read_csv(os.path.join(CLEAN, f"task_{task}.csv"), low_memory=False, encoding="utf-8-sig"))
    feats = feature_cols(df)
    scenes = ["IP","OP","ER","other"]
    shap_mat = {}   # feature -> list of mean|SHAP| per scene-model
    sign_mat = {}   # feature -> list of SHAP sign per scene-model
    for sc in scenes:
        tr = df[df["ward_raw"] != sc]
        m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                               min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                               random_state=42, verbose=-1)
        m.fit(tr[feats], tr["label"])
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(df[feats])
        sv = sv[1] if isinstance(sv, list) else sv
        for i, f in enumerate(feats):
            shap_mat.setdefault(f, []).append(np.abs(sv[:, i]).mean())
            sign_mat.setdefault(f, []).append(np.sign(sv[:, i]).mean())
    # 稳定性指标
    rows = []
    for f in feats:
        imp = np.array(shap_mat[f]); sgn = np.array(sign_mat[f])
        rows.append(dict(task=task, feature=f,
                         mean_shap=imp.mean(),
                         rank_stability=np.nan,   # 下面统一算
                         sign_consistency=float((np.sign(sgn) == np.sign(sgn.mean())).mean() if sgn.mean() != 0 else 0),
                         is_suspect=int(f in SUSPECTS)))
    rdf = pd.DataFrame(rows)
    # 排序稳定性: 跨场景重要性排序的 Spearman
    rank_df = pd.DataFrame({sc: pd.Series(shap_mat[f]).rank(ascending=False).values for f, sc in zip(feats, scenes)}) if False else None
    # 直接: 每场景的重要性向量做 Spearman
    imp_vecs = np.array([shap_mat[f] for f in feats])   # (n_feat, n_scene)
    rho_sum = 0; n_pair = 0
    for a in range(len(scenes)):
        for b in range(a + 1, len(scenes)):
            rho, _ = spearmanr(imp_vecs[:, a], imp_vecs[:, b])
            if not np.isnan(rho): rho_sum += rho; n_pair += 1
    rdf["rank_stability"] = rho_sum / max(n_pair, 1)
    rdf = rdf.sort_values("mean_shap", ascending=False)
    stable = rdf[(rdf["rank_stability"] >= 0.7) & (rdf["sign_consistency"] >= 0.75)]
    p(f"\n===== {task} =====")
    p(f"场景间 Spearman 均值: {rdf['rank_stability'].iloc[0]:.3f}")
    p("[候选稳定] " + " > ".join(f"{r.feature}({r.mean_shap:.3f}, ρ={r.rank_stability:.2f})" for r in stable.head(12).itertuples()))
    p("[疑似捷径(SHAP高但方向不稳)] " + " > ".join(
        f"{r.feature}({r.mean_shap:.3f}, 方向一致率={r.sign_consistency:.2f})"
        for r in rdf[(rdf['mean_shap'] >= rdf['mean_shap'].median()) & (rdf['sign_consistency'] < 0.75)].head(6).itertuples()))
    all_rows.append(rdf)

pd.concat(all_rows, ignore_index=True).to_csv(os.path.join(OUT, "stable_features.csv"), index=False)
p("\n已存 stable_features.csv")
report.close()
print("完成", flush=True)
