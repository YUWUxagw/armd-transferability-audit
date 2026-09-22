# -*- coding: utf-8 -*-
"""
Step 39 — Table 1 (三站队列特征) + Table 2 (跨系统转移矩阵)
从冻结数据集与 final results 生成, 数字全部可追溯
输出: E:\ARMD\paper\tables\table1.csv + table2.csv (含 LaTeX 就绪列名)
"""
import os
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
R3 = os.path.join(BASE, "05_源数据", "phase3")
OUT = os.path.join(BASE, "03_表格")
os.makedirs(OUT, exist_ok=True)

# ========== Table 1 ==========
TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SITES = ["MGB","Stanford","UTSW"]
SITE_LABEL = {"MGB":"MGB","Stanford":"Stanford","UTSW":"UTSW"}
rows = []
for s in SITES:
    # Superset
    base = pd.read_csv(os.path.join(CLEAN, s, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    n_cx = len(base); n_pat = base["anon_id"].nunique()
    rows.append(dict(Site=SITE_LABEL[s], Task="PA superset", n_cultures=n_cx, n_patients=n_pat,
                     R_rate=f"{base['label'].mean():.3f}" if "label" in base.columns else "—"))
    for t in TASKS:
        df = pd.read_csv(os.path.join(CLEAN, s, f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
        nR = int(df["label"].sum())
        n = len(df)
        rows.append(dict(Site=SITE_LABEL[s], Task=t.upper(), n_cultures=n, n_patients=df["anon_id"].nunique(),
                         nR=nR, R_rate=f"{df['label'].mean():.3f}"))

t1 = pd.DataFrame(rows)
# Pivot: Site as columns, Task as rows
t1.to_csv(os.path.join(OUT, "table1_long.csv"), index=False)
# Wide
piv = t1.pivot(index="Task", columns="Site", values=["n_cultures","R_rate"])
piv.to_csv(os.path.join(OUT, "table1_wide.csv"))
print("Table 1 (wide):")
print(piv.to_string())
print()

# ========== Table 2 ==========
tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
cv = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "cv_internal.csv"))
# Get internal AUROC per task (lgb mean)
internal = cv[cv["model"]=="lgb"].groupby("task")["auroc"].mean().to_dict()

t2_rows = []
for _, r in tm.iterrows():
    t2_rows.append(dict(
        Task=r["task"].upper(),
        Direction=f"{r['src']}→{r['tgt']}",
        n_test=int(r["n_test"]),
        AUROC=f"{r['auroc']:.3f}",
        Gap=f"{r['transfer_gap']:+.3f}",
        AUPRC=f"{r['auprc']:.3f}",
        Brier=f"{r['brier']:.3f}",
        Calib_Int=f"{r['calib_int']:+.2f}",
        Calib_Slope=f"{r['calib_slope']:.2f}",
        Internal_AUROC=f"{internal.get(r['task'], 0):.3f}"
    ))
t2 = pd.DataFrame(t2_rows)
t2.to_csv(os.path.join(OUT, "table2.csv"), index=False)
print("Table 2:")
for _, r in t2.iterrows():
    print(f"  {r['Task']:6s} {r['Direction']:12s} AUROC {r['AUROC']} (gap {r['Gap']}) "
          f"Brier {r['Brier']} Int {r['Calib_Int']} Slope {r['Calib_Slope']} "
          f"Internal {r['Internal_AUROC']}")

print(f"\nTables saved to {OUT}")
