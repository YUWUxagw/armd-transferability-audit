# -*- coding: utf-8 -*-
"""抽查关键特征漂移比值"""
import pandas as pd

d = pd.read_csv(r"E:\ARMD\results\phase3\feature_drift.csv")
for tgt in ["Stanford", "UTSW"]:
    sub = d[d["tgt"] == tgt]
    print(f"=== MGB→{tgt} 关键特征漂移 (跨任务) ===")
    for f in ["prior_pa_res_levofloxacin", "adi", "comorb_count",
              "abx_carbapenem_0_30", "prior_pa_res_tobramycin", "prior_pseudo"]:
        r = sub[sub["feature"] == f]
        if len(r):
            print(f"  {f}: 比值均值 {r['ratio'].mean():.2f} | 各任务: {[round(x, 2) for x in r['ratio'].tolist()]}")
