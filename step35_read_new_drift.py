# -*- coding: utf-8 -*-
"""读 adi 剔除后新漂移表关键数字"""
import pandas as pd

d = pd.read_csv(r"E:\ARMD\results\phase3\feature_drift.csv")
for tgt in ["Stanford", "UTSW"]:
    sub = d[d["tgt"] == tgt]
    st = sub[sub["in_stable_set"] == 1]
    print(f"[{tgt}] 稳定均值 {st['ratio'].mean():.2f}")
    for f in ["adi", "comorb_count", "prior_pa_res_levofloxacin", "abx_carbapenem_0_30",
              "prior_pa_res_tobramycin", "abx_antipseudomonal_bl_0_30",
              "prior_pa_res_piperacillin_tazobactam", "abx_glycopeptide_0_30"]:
        r = sub[sub["feature"] == f]
        if len(r):
            print(f"  {f}: {r['ratio'].mean():.2f} {[round(x, 2) for x in r['ratio'].tolist()]}")
