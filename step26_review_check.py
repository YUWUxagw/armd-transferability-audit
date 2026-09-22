# -*- coding: utf-8 -*-
"""
Step 26 — 草稿数字核对 (Methods_Results_v1 的全部可编程核对点 vs 结果文件)
输出: audit_out/report_step26_review.txt (权威数字, 供逐项比对)
"""
import os
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
R2 = os.path.join(BASE, "05_源数据", "phase2")
R3 = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(BASE, "audit_out", "report_step26_review.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SITES = ["MGB","Stanford","UTSW"]

# 1. 队列数字
p("=== 1. 队列 (任务级: n 培养, R率) ===")
for s in SITES:
    for t in TASKS:
        df = pd.read_csv(os.path.join(CLEAN, s, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        p(f"  {s} {t}: n={len(df):,} R率={df['label'].mean():.3f} 患者={df['anon_id'].nunique():,}")

# 2. 内部 CV
p("\n=== 2. 内部 CV (cv_internal.csv) ===")
cv = pd.read_csv(os.path.join(R2, "cv_internal.csv"))
for t in TASKS:
    for m in ["lgb","lr"]:
        s = cv[(cv["task"]==t)&(cv["model"]==m)]
        p(f"  {t} {m}: AUROC {s['auroc'].mean():.3f}±{s['auroc'].std():.3f} AUPRC {s['auprc'].mean():.3f}")

# 3. 场景梯度
p("\n=== 3. 场景梯度 ===")
sg = pd.read_csv(os.path.join(R2, "scene_gradient.csv"))
for t in TASKS:
    s = sg[sg["task"]==t]
    p(f"  {t}: " + " | ".join(f"{r.scene}:{r.auroc:.3f}(n={r.n_test})" for r in s.itertuples()))

# 4. 跨系统
p("\n=== 4. 跨系统转移矩阵 ===")
tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
for t in TASKS:
    s = tm[tm["task"]==t]
    p(f"  {t}: " + " | ".join(f"{r.src}→{r.tgt}:{r.auroc:.3f}({r.transfer_gap:+.3f}) Brier {r.brier:.3f} 截距 {r.calib_int:+.2f} 斜率 {r.calib_slope:.2f}" for r in s.itertuples()))

# 5. 特征漂移 (关键特征)
p("\n=== 5. 特征漂移 (feature_drift.csv 均值) ===")
fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
KEYS = ["adi","comorb_count","prior_pa_res_levofloxacin","abx_glycopeptide_0_30",
        "abx_carbapenem_0_30","prior_pa_res_piperacillin_tazobactam",
        "abx_antipseudomonal_bl_0_30","prior_pa_res_tobramycin",
        "proc_urinary_cath","proc_dialysis"]
for tgt in ["Stanford","UTSW"]:
    d = fd[fd["tgt"]==tgt]
    st = d[d["in_stable_set"]==1]; ns = d[d["in_stable_set"]==0]
    p(f"  [{tgt}] 稳定均值 {st['ratio'].mean():.2f} 非稳定 {ns['ratio'].mean():.2f}")
    for f in KEYS:
        r = d[d["feature"]==f]
        if len(r): p(f"    {f}: {r['ratio'].mean():.2f} 各任务 {[round(x,2) for x in r['ratio'].tolist()]}")

# 6. 偏移分解
p("\n=== 6. 偏移分解 ===")
sd = pd.read_csv(os.path.join(R3, "shift_decomposition.csv"))
for _, r in sd.iterrows():
    p(f"  {r['task']}→{r['tgt']}: 标签偏移 {r['label_shift_pp']:+.1f}pp 协变量AUC {r['covariate_auc']:.3f} "
      f"AUROC {r['base_auc']:.3f} Brier {r['brier_raw']:.3f}→{r['brier_label_adj']:.3f} "
      f"截距 {r['calib_int_raw']:+.2f}→{r['calib_int_label_adj']:+.2f} SMDtop {r['top_smd']}")

# 7. 假设三
p("\n=== 7. 稳定子集 (stable_subset.csv) ===")
ss = pd.read_csv(os.path.join(R3, "stable_subset.csv"))
for _, r in ss.iterrows():
    p(f"  {r['task']}→{r['tgt']}: 全 {r['full_external']:.3f}(缺口{r['full_gap']:+.3f}) 子集 {r['subset_external']:.3f}(缺口{r['subset_gap']:+.3f}) 缺口变化{r['gap_reduction']:+.3f} CI {r['boot_ci']}")

# 8. DCA
p("\n=== 8. DCA (dca.csv, full 模型) ===")
dc = pd.read_csv(os.path.join(R3, "dca.csv"))
dc = dc[dc["model"]=="full"]
for t in TASKS:
    for g in ["Stanford","UTSW"]:
        s = dc[(dc["task"]==t)&(dc["tgt"]==g)]
        b = s.loc[s["net_benefit"].idxmax()]
        nb20 = s[s["threshold"]==0.20]["net_benefit"].iloc[0] if (s["threshold"]==0.20).any() else None
        p(f"  {t}→{g}: 最大净获益 {b['net_benefit']:.4f}@t={b['threshold']:.2f} | t=0.20 {nb20}")

rep.close()
print("完成", flush=True)
