# -*- coding: utf-8 -*-
"""量测: PA 患者全量培养规模 (选择修复方案)"""
import pandas as pd

mc = pd.read_csv(r"E:\ARMD\ARMD-MGB\microbiology_cohort_deid_tj_updated.csv",
                 usecols=["anon_id", "order_proc_id_coded", "order_time_jittered_utc_shifted"],
                 low_memory=False, encoding="utf-8-sig")
pa_ids = set(pd.read_csv(r"E:\ARMD\data\clean\MGB\site_base_v3.csv", usecols=["anon_id"])["anon_id"])
print("PA 患者数:", len(pa_ids))
sub = mc[mc["anon_id"].isin(pa_ids)]
print("PA 患者的全部培养行:", f"{len(sub):,}")
cult = sub.drop_duplicates("order_proc_id_coded")
print("PA 患者的全部唯一培养:", f"{len(cult):,}")
cnt = cult.groupby("anon_id").size()
print("每患者培养数: 中位", cnt.median(), "| P90", cnt.quantile(.9), "| max", cnt.max())
pairs = int((cnt * (cnt - 1) / 2).sum())
print("全量培养对估算:", f"{pairs:,}")
abx = pd.read_csv(r"E:\ARMD\ARMD-MGB\prior_abx_deid_tj.csv", usecols=["anon_id"],
                  low_memory=False, encoding="utf-8-sig")
print("PA 患者的 prior_abx 行:", f"{int(abx[abx['anon_id'].isin(pa_ids)].shape[0]):,}")
