# -*- coding: utf-8 -*-
"""确认 proc 特征在三站基座的分布 (漂移 0.00 疑点)"""
import pandas as pd

for site in ["MGB", "Stanford", "UTSW"]:
    b = pd.read_csv(rf"E:\ARMD\data\clean\{site}\site_base_v3.csv", low_memory=False, encoding="utf-8-sig")
    print(f"{site}: 基座 {len(b)} 培养 | "
          f"proc_urinary_cath={int((b['proc_urinary_cath']==1).sum())} "
          f"proc_dialysis={int((b['proc_dialysis']==1).sum())} "
          f"proc_mechvent={int((b['proc_mechvent']==1).sum())} "
          f"proc_surgery={int((b['proc_surgery']==1).sum())} "
          f"nh_30d={int((b['nh_30d']==1).sum())}")
