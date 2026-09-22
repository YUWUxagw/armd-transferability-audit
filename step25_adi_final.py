# -*- coding: utf-8 -*-
"""核对最终 feature_drift.csv 中 adi 及关键特征比值 (草稿用)"""
import pandas as pd

d = pd.read_csv(r"E:\ARMD\results\phase3\feature_drift.csv")
for tgt in ["Stanford", "UTSW"]:
    sub = d[d["tgt"] == tgt]
    print(f"=== MGB→{tgt} ===")
    for f in ["adi", "comorb_count", "prior_pa_res_levofloxacin", "abx_carbapenem_0_30",
              "prior_pa_res_tobramycin", "abx_antipseudomonal_bl_0_30", "prior_pa_res_piperacillin_tazobactam",
              "abx_glycopeptide_0_30"]:
        r = sub[sub["feature"] == f]
        if len(r):
            print(f"  {f}: 均值 {r['ratio'].mean():.2f} | 各任务 {[round(x, 2) for x in r['ratio'].tolist()]}")
