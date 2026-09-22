# -*- coding: utf-8 -*-
"""验证: 三站分类列首次出现顺序 vs 编码 (pandas cat.codes 顺序假设)"""
import pandas as pd

CLEAN = r"E:\ARMD\data\clean"

for task in ["meropenem", "ciprofloxacin"]:
    print(f"=== {task} ===")
    for site in ["MGB", "Stanford", "UTSW"]:
        df = pd.read_csv(rf"{CLEAN}\{site}\task_{task}.csv", low_memory=False, encoding="utf-8-sig")
        for c in ["age_bin", "ward_h", "specimen"]:
            s = df[c].astype(str)
            codes = s.astype("category").cat.codes
            order = list(dict.fromkeys(s.tolist()))
            print(f"  [{site}] {c}: 首现顺序 {order} | 编码 {dict(zip(order, [int(codes.iloc[s[s==v].index[0]]) for v in order]))}")
