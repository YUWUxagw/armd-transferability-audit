# -*- coding: utf-8 -*-
"""读 DCA 代表性数字 (Results 草稿用)"""
import pandas as pd

d = pd.read_csv(r"E:\ARMD\results\phase3\dca.csv")
d = d[d["model"] == "full"]
out = []
for t in ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]:
    for g in ["Stanford", "UTSW"]:
        s = d[(d["task"] == t) & (d["tgt"] == g)]
        best = s.loc[s["net_benefit"].idxmax()]
        nb20 = s[s["threshold"] == 0.20]["net_benefit"].iloc[0] if (s["threshold"] == 0.20).any() else None
        # 全治疗基准: 净获益 = 患病率 (t=0)
        prev = None
        out.append(f"{t}->{g}: 最大净获益 {best['net_benefit']:.4f} @t={best['threshold']:.2f} | t=0.20 净获益 {nb20}")
print("\n".join(out))
