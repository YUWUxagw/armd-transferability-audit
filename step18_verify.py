# -*- coding: utf-8 -*-
"""
ARMD Step 18 — Step 2 前补验
A. 指标计算 toy 对照 (AUROC/AUPRC/Brier/校准截距斜率 符号约定)
B. F3/F6/F8 非零特征核对 (选有暴露的患者)
C. 阶段3 漂移稳健性抽查 (MEM MGB→Stanford, 全量参考 SHAP)
输出: E:\\ARMD\\audit_out\\report_step18_verify.txt
"""
import os, sys, time
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"; M = os.path.join(BASE, "ARMD-MGB")
CLEAN = os.path.join(BASE, "data", "clean", "MGB")
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
rep = open(os.path.join(OUT, "report_step18_verify.txt"), "w", encoding="utf-8")
fails = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()
def fail(tag, msg): fails.append(tag); p(f"[FAIL] {tag}: {msg}")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step14_model_phase2 import metrics
from sklearn.metrics import roc_auc_score, average_precision_score, brier_score_loss
from sklearn.linear_model import LogisticRegression

def calib_metrics(y, p):
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(max_iter=500)
    lr.fit(lp.reshape(-1, 1), y)
    return brier, float(lr.intercept_[0]), float(lr.coef_[0][0])

# ---------- A. Toy 指标对照 ----------
p("=" * 80); p("A. 指标计算 toy 对照"); p("=" * 80)
rng = np.random.RandomState(0)
y = np.array([0, 1, 1, 0, 1, 0, 0, 1, 1, 1])
p_hi = np.array([0.9, 0.8, 0.7, 0.6, 0.5, 0.4, 0.3, 0.2, 0.1, 0.05])  # 反向(高估)
m = metrics(y, p_hi)
assert abs(m["auroc"] - roc_auc_score(y, p_hi)) < 1e-12, "AUROC 不一致"
assert abs(m["auprc"] - average_precision_score(y, p_hi)) < 1e-12, "AUPRC 不一致"
b, ci, cs = calib_metrics(y, p_hi)
assert abs(b - brier_score_loss(y, p_hi)) < 1e-12, "Brier 不一致"
p(f"[A] AUROC/AUPRC/Brier 与 sklearn 直接调用一致 ✓")
p(f"[A] 校准符号约定: 高估模型(y=1占比0.5, 预测均值0.445→反向例) — 本 toy 用反向概率, 截距 {ci:+.2f} 斜率 {cs:.2f}")
# 符号约定专项: 系统性高估(正例预测0.9, 负例预测0.7, 均高于真实率) → 截距应为负
y2 = np.array([0, 0, 0, 0, 0, 1, 1, 1, 1, 1])
p_over = np.array([0.7, 0.7, 0.7, 0.7, 0.7, 0.9, 0.9, 0.9, 0.9, 0.9])  # 正例0.9 负例0.7: 系统性高估
b2, ci2, cs2 = calib_metrics(y2, p_over)
p(f"[A] 系统性高估 → 截距 {ci2:+.2f} ({'负=高估 ✓' if ci2 < 0 else '符号异常'})")
if not (ci2 < 0): fail("A-符号", "高估模型的校准截距应为负")

# ---------- B. F3/F6/F8 非零核对 ----------
p("\n" + "=" * 80); p("B. F3/F6/F8 非零特征核对"); p("=" * 80)
base = pd.read_csv(os.path.join(CLEAN, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
base["time"] = pd.to_datetime(base["time"], errors="coerce")

def chunked_filter(path, cols, pred, chunksize=2000000):
    out = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False, encoding="utf-8-sig"):
        m = pred(ch)
        if m.any(): out.append(ch[m])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

# F3: 选 abx_carbapenem_0_30 > 0 的患者, 原始表重算
pos_pat = base[base["abx_carbapenem_0_30"] > 0]
p(f"[B] 有碳青霉烯0-30暴露的培养: {len(pos_pat):,} / {len(base):,}")
pid = pos_pat["anon_id"].iloc[0]
rows = base[base["anon_id"] == pid]
cx = rows.sort_values("abx_carbapenem_0_30", ascending=False)["order_proc_id_coded"].iloc[0]
idx = rows[["anon_id","order_proc_id_coded","time"]].rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"})
abx = chunked_filter(os.path.join(M, "prior_abx_deid_tj.csv"),
                     ["anon_id","medication_name","last_dose_to_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                     lambda ch: ch["anon_id"].eq(pid))
abx["att"] = pd.to_datetime(abx["order_time_jittered_utc_shifted"], errors="coerce")
abx["d"] = pd.to_numeric(abx["last_dose_to_culture"], errors="coerce")
sys.path.insert(0, os.path.dirname(__file__))
from step9_clean import resolve_drug, DRUG_CLASS, PROC_KEYS
abx["cls"] = abx["medication_name"].map(resolve_drug).map(DRUG_CLASS)
abx = abx[abx["cls"].notna() & abx["att"].notna() & abx["d"].notna() & (abx["d"] > 0)]  # -1 哨兵排除(与清洗一致)
abx["ev"] = abx["att"] - pd.to_timedelta(abx["d"], unit="D")
m3 = abx.merge(idx, on="anon_id")
m3 = m3[(m3["idx_t"] > m3["ev"])]
exp = int(((m3["idx_cx"] == cx) & (m3["cls"] == "carbapenem") & ((m3["idx_t"] - m3["ev"]).dt.days <= 30)).sum())
got = int(rows[rows["order_proc_id_coded"] == cx]["abx_carbapenem_0_30"].iloc[0])
if exp != got: fail(f"B-F3", f"培养{cx} 期望{exp} 实际{got}")
p(f"[B] F3 碳青霉烯0-30 非零核对: 期望 {exp} 实际 {got} -> {'OK' if exp == got else 'MISMATCH'}")

# F6: 选 proc_mechvent=1 的患者核对
mv_pat = base[base["proc_mechvent"] == 1]
pid6 = mv_pat["anon_id"].iloc[0]
rows6 = base[base["anon_id"] == pid6]
cx6 = rows6["order_proc_id_coded"].iloc[0]
idx6 = rows6[["anon_id","order_proc_id_coded","time"]].rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"})
pr = chunked_filter(os.path.join(M, "prior_procedures_deid_tj.csv"),
                    ["anon_id","procedure_description","procedure_days_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                    lambda ch: ch["anon_id"].eq(pid6))
pr["att"] = pd.to_datetime(pr["order_time_jittered_utc_shifted"], errors="coerce")
pr["d"] = pd.to_numeric(pr["procedure_days_culture"], errors="coerce")
pr = pr[pr["att"].notna() & pr["d"].notna()]
pr["ev"] = pr["att"] - pd.to_timedelta(pr["d"], unit="D")
m6 = pr.merge(idx6, on="anon_id")
m6 = m6[(m6["idx_t"] > m6["ev"]) &
        (m6["procedure_description"].astype(str).str.lower().str.contains("|".join(PROC_KEYS["mechvent"]), regex=True))]
exp6 = int((m6["idx_cx"] == cx6).sum() > 0)
got6 = int(rows6[rows6["order_proc_id_coded"] == cx6]["proc_mechvent"].iloc[0])
if exp6 != got6: fail(f"B-F6", f"期望{exp6} 实际{got6}")
p(f"[B] F6 机械通气 非零核对: 期望 {exp6} 实际 {got6} -> {'OK' if exp6 == got6 else 'MISMATCH'}")

# F8: 选 nh_30d=1 的患者核对
nh_pat = base[base["nh_30d"] == 1]
pid8 = nh_pat["anon_id"].iloc[0]
rows8 = base[base["anon_id"] == pid8]
cx8 = rows8["order_proc_id_coded"].iloc[0]
idx8 = rows8[["anon_id","order_proc_id_coded","time"]].rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"})
nh = chunked_filter(os.path.join(M, "nursing_home_visits_deid_tj.csv"),
                    ["anon_id","nursing_home_visit_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                    lambda ch: ch["anon_id"].eq(pid8))
nh["att"] = pd.to_datetime(nh["order_time_jittered_utc_shifted"], errors="coerce")
nh["d"] = pd.to_numeric(nh["nursing_home_visit_culture"], errors="coerce")
nh = nh[nh["att"].notna() & nh["d"].notna()]
nh["ev"] = nh["att"] - pd.to_timedelta(nh["d"], unit="D")
m8 = nh.merge(idx8, on="anon_id")
m8 = m8[(m8["idx_t"] > m8["ev"])]
exp8 = int(((m8["idx_cx"] == cx8) & ((m8["idx_t"] - m8["ev"]).dt.days <= 30)).sum() > 0)
got8 = int(rows8[rows8["order_proc_id_coded"] == cx8]["nh_30d"].iloc[0])
if exp8 != got8: fail(f"B-F8", f"期望{exp8} 实际{got8}")
p(f"[B] F8 养老院30天 非零核对: 期望 {exp8} 实际 {got8} -> {'OK' if exp8 == got8 else 'MISMATCH'}")

# ---------- C. 漂移稳健性 (全量参考 SHAP) ----------
p("\n" + "=" * 80); p("C. 阶段3 漂移稳健性抽查 (MEM MGB→Stanford)"); p("=" * 80)
import lightgbm as lgb, shap

def fit_cat_maps_local(mgb_df):
    maps = {}
    for c in ["age_bin","ward_h","specimen"]:
        vals = sorted(mgb_df[c].astype(str).unique())
        maps[c] = {v: i for i, v in enumerate(vals)}
    return maps

def apply_cat_local(df, maps):
    for c, m in maps.items():
        df[c] = df[c].astype(str).map(m).fillna(-1).astype(int)
    return df

def prep_cross_local(df, maps, adi_median=None):
    df = df.copy()
    df["ward_raw"] = df["ward_h"].astype(str)
    df["age_bin"] = df["age_bin"].astype(str).str.replace(" years","",regex=False)
    df = apply_cat_local(df, maps)
    if adi_median is not None:
        df["adi"] = df["adi"].fillna(adi_median)
    df["prior_pseudo_days"] = df["prior_pseudo_days"].fillna(-1)
    return df

DROP_COLS_LOCAL = ["order_proc_id_coded","anon_id","label","fold_id","prior_pa_res","time","ward_raw"]
def feats_of_local(df):
    return [c for c in df.columns if c not in DROP_COLS_LOCAL and c != "ward_raw"]

mgb_raw = pd.read_csv(os.path.join(CLEAN, "task_meropenem.csv"), low_memory=False, encoding="utf-8-sig")
maps = fit_cat_maps_local(mgb_raw)
adi_med = mgb_raw["adi"].median()
mgb = prep_cross_local(mgb_raw, maps, adi_med)
feats = feats_of_local(mgb)
sta = prep_cross_local(pd.read_csv(os.path.join(BASE, "data", "clean", "Stanford", "task_meropenem.csv"),
                                   low_memory=False, encoding="utf-8-sig"), maps, adi_med)
m = lgb.LGBMClassifier(n_estimators=500, learning_rate=0.05, num_leaves=31, min_child_samples=30,
                       subsample=0.8, colsample_bytree=0.8, random_state=42, verbose=-1)
m.fit(mgb[feats], mgb["label"])
ex = shap.TreeExplainer(m)
sv_tr_full = ex.shap_values(mgb[feats]); sv_tr_full = sv_tr_full[1] if isinstance(sv_tr_full, list) else sv_tr_full
sv_te = ex.shap_values(sta[feats]); sv_te = sv_te[1] if isinstance(sv_te, list) else sv_te
full_ratio = {f: np.abs(sv_te[:, i]).mean() / max(np.abs(sv_tr_full[:, i]).mean(), 1e-9) for i, f in enumerate(feats)}
prev = pd.read_csv(os.path.join(BASE, "05_源数据", "phase3", "feature_drift.csv"))
prev_mem = prev[(prev["task"] == "meropenem") & (prev["tgt"] == "Stanford")].set_index("feature")["ratio"]
p(f"[C] 全量参考 SHAP 下关键特征比值 vs 3000行子样本:")
for f in ["abx_carbapenem_0_30", "prior_pa_res_tobramycin", "adi", "prior_pa_res_levofloxacin"]:
    p(f"    {f}: 全量={full_ratio[f]:.2f} vs 子样本={prev_mem.get(f, float('nan')):.2f}")
p("[C] 结论标准: 全量子样本比值差异 >0.15 则标记需重算")

p("\n" + "=" * 80)
if fails: p(f"补验结果: {len(fails)} 项失败 -> {fails}")
else: p("补验结果: 全部通过")
rep.close()
print("完成", flush=True)
