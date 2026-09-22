# -*- coding: utf-8 -*-
"""
ARMD 阶段2 — MGB 建模 (5 主任务)
设置: ① 内部 5 折患者级 CV  ② 场景梯度迁移(留一场景)  ③ SHAP 归因  ④ 消融(预登记捷径)
模型: LightGBM + LogisticRegression(缩放后)
输出: E:\\ARMD\\results\\phase2\\* (报告 txt + 汇总 csv)
"""
import os, time, warnings
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean", "MGB")
OUT = os.path.join(BASE, "05_源数据", "phase2")
os.makedirs(OUT, exist_ok=True)

PRIMARY = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
# 预登记疑似捷径特征 (消融)
SUSPECTS = ["mucoid","ward_h","specimen","icu","adi"]
# 主模型排除特征 (2026-08-07 评审驱动: adi 三站标尺不一致(MGB原始分数 vs S/U百分位),
# 内部无贡献(剔除Δ≤0.007)但跨系统缺口全部收窄(+0.002~+0.062) → 从主模型剔除, 留作敏感性)
EXCLUDE_FEATURES = ["adi"]
DROP_COLS = ["order_proc_id_coded","anon_id","label","fold_id","prior_pa_res","time"]

def prep(df):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)              # 场景梯度用原始值
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years","",regex=False)
    # 编码 (编码后保留为特征)
    for c in ["age_bin","ward_h","specimen"]:
        df[c] = df[c].astype("category").cat.codes
    # 缺失填充
    df["adi"] = df["adi"].fillna(df["adi"].median())
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

def feature_cols(df):
    return [c for c in df.columns if c not in DROP_COLS and c != "ward_raw" and c not in EXCLUDE_FEATURES]

# 跨站编码: MGB 拟合映射, 目标侧统一使用 (2026-08-07 修复: 独立 prep 的
# cat.codes 在类别值集不一致时(如 S/U 无 ER ward)编码错位, 污染跨站比较)
def fit_cat_maps(df):
    maps = {}
    for c in ["age_bin","ward_h","specimen"]:
        vals = sorted(df[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps

def prep_cross(df, maps, adi_med):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years","",regex=False)
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).astype(int)
        if (df[c] < 0).any():
            raise ValueError(f"编码错位: {c} 存在未映射值 (映射必须在原始数据上拟合)")
    df["adi"] = df["adi"].fillna(adi_med)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

def metrics(y, p):
    from sklearn.metrics import roc_auc_score, average_precision_score
    return dict(auroc=roc_auc_score(y, p), auprc=average_precision_score(y, p))

from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
import lightgbm as lgb

def train_lgb(X, y, Xv, yv):
    m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31,
                           min_child_samples=30, subsample=0.8, colsample_bytree=0.8,
                           random_state=42, verbose=-1)
    # 2026-09-03 修复 (汇总意见 K26): 移除 eval_set early stopping。
    # 旧版把 CV 验证折既用作 early stopping 监视集又用作评估集 (selection-on-eval 泄漏)。
    # 诊断: ES 折级 best_iter 45-93, 折级 AUROC 被抬高 0.010-0.041, 池化抬升 0.023-0.032。
    # 现统一为固定 500 树, 与 step16 跨系统训练制度一致; Xv/yv 保留仅维持调用签名。
    m.fit(X, y)
    return m

def train_lr(X, y):
    sc = StandardScaler().fit(X)
    m = LogisticRegression(max_iter=2000, C=0.5, random_state=42)
    m.fit(sc.transform(X), y)
    return m, sc

def cv_internal(df, feats, model="lgb"):
    rows = []
    for f in sorted(df["fold_id"].unique()):
        tr, va = df[df["fold_id"] != f], df[df["fold_id"] == f]
        if model == "lgb":
            m = train_lgb(tr[feats], tr["label"], va[feats], va["label"])
            p = m.predict_proba(va[feats])[:, 1]
        else:
            m, sc = train_lr(tr[feats], tr["label"]); p = m.predict_proba(sc.transform(va[feats]))[:, 1]
        rows.append({"fold": int(f), **metrics(va["label"], p)})
    return pd.DataFrame(rows)

def scene_gradient(df, feats, model="lgb"):
    rows = []
    for sc in ["IP","OP","ER","other"]:
        tr = df[df["ward_raw"] != sc]; va = df[df["ward_raw"] == sc]
        if va["label"].nunique() < 2 or len(va) < 50:
            rows.append({"scene": sc, "n_test": len(va), "auroc": np.nan, "auprc": np.nan}); continue
        if model == "lgb":
            m = train_lgb(tr[feats], tr["label"], va[feats], va["label"])
            p = m.predict_proba(va[feats])[:, 1]
        else:
            m, sca = train_lr(tr[feats], tr["label"]); p = m.predict_proba(sca.transform(va[feats]))[:, 1]
        rows.append({"scene": sc, "n_test": len(va), **metrics(va["label"], p)})
    return pd.DataFrame(rows)

def main():
    report = open(os.path.join(OUT, "report_phase2.txt"), "w", encoding="utf-8")
    def p(*a):
        line = " ".join(str(x) for x in a)
        print(line, flush=True); report.write(line + "\n")
    all_cv, all_sg, all_abl = [], [], []
    for task in PRIMARY:
        t0 = time.time()
        fp = os.path.join(CLEAN, f"task_{task}.csv")
        if not os.path.exists(fp):
            p(f"[{task}] 文件缺失，跳过"); continue
        df = prep(pd.read_csv(fp, low_memory=False, encoding="utf-8-sig"))
        feats = feature_cols(df)
        p(f"\n===== {task} | n={len(df):,} | R率 {df['label'].mean():.3f} | 特征 {len(feats)} =====")

        # ① 内部 CV
        cv = cv_internal(df, feats)
        p(f"[内部CV] LightGBM AUROC {cv['auroc'].mean():.3f}±{cv['auroc'].std():.3f} | AUPRC {cv['auprc'].mean():.3f}")
        cv_lr = cv_internal(df, feats, model="lr")
        p(f"[内部CV] LR        AUROC {cv_lr['auroc'].mean():.3f}±{cv_lr['auroc'].std():.3f}")
        cv["task"] = task; cv["model"] = "lgb"; all_cv.append(cv)
        cv_lr["task"] = task; cv_lr["model"] = "lr"; all_cv.append(cv_lr)

        # ② 场景梯度 (LightGBM)
        sg = scene_gradient(df, feats)
        p(f"[场景梯度] " + " | ".join(f"{r.scene}:AUROC {r.auroc:.3f}(n={r.n_test})" for r in sg.itertuples()))
        sg["task"] = task; all_sg.append(sg)

        # ③ SHAP 归因 (全量训练)
        m = train_lgb(df[feats], df["label"], df[feats], df["label"])
        import shap
        ex = shap.TreeExplainer(m)
        sv = ex.shap_values(df[feats])[1] if isinstance(ex.shap_values(df[feats]), list) else ex.shap_values(df[feats])
        imp = pd.DataFrame({"feature": feats, "mean_abs_shap": np.abs(sv).mean(axis=0)}).sort_values("mean_abs_shap", ascending=False)
        p("[SHAP top10] " + " > ".join(f"{r.feature}({r.mean_abs_shap:.4f})" for r in imp.head(10).itertuples()))
        imp.to_csv(os.path.join(OUT, f"shap_{task}.csv"), index=False)

        # ④ 消融 (预登记捷径, LightGBM 内部 CV)
        abl_rows = []
        for s in SUSPECTS:
            if s not in feats: continue
            feats_abl = [f for f in feats if f != s]
            a = cv_internal(df, feats_abl)["auroc"].mean()
            abl_rows.append(dict(removed=s, auroc=a))
        abl = pd.DataFrame(abl_rows)
        abl["baseline"] = cv["auroc"].mean()
        p("[消融] " + " | ".join(f"-{r.removed}: {r.auroc:.3f}" for r in abl.itertuples()))
        abl["task"] = task; all_abl.append(abl)
        p(f"  耗时 {time.time()-t0:.0f}s")

    pd.concat(all_cv, ignore_index=True).to_csv(os.path.join(OUT, "cv_internal.csv"), index=False)
    pd.concat(all_sg, ignore_index=True).to_csv(os.path.join(OUT, "scene_gradient.csv"), index=False)
    pd.concat(all_abl, ignore_index=True).to_csv(os.path.join(OUT, "ablation.csv"), index=False)
    p("\n汇总已存 results/phase2/")
    report.close()

if __name__ == "__main__":
    main()
    print("完成", flush=True)
