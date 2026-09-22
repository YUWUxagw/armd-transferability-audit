# -*- coding: utf-8 -*-
"""查证: proc_mechvent 是否全零 + MGB 操作描述词表"""
import pandas as pd

base = pd.read_csv(r"E:\ARMD\data\clean\MGB\site_base_v3.csv", low_memory=False, encoding="utf-8-sig")
for c in ["proc_mechvent", "proc_cvc", "proc_urinary_cath", "proc_dialysis", "proc_surgery", "nh_30d"]:
    print(f"{c}: 阳性培养数 {int((base[c] == 1).sum()):,} / {len(base):,}")

pr = pd.read_csv(r"E:\ARMD\ARMD-MGB\prior_procedures_deid_tj.csv",
                 usecols=["procedure_description"], nrows=300000, low_memory=False, encoding="utf-8-sig")
print("\nMGB prior_procedures 描述 top30:")
print(pr["procedure_description"].value_counts().head(30).to_string())
