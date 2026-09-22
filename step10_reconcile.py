# -*- coding: utf-8 -*-
"""
ARMD Step 10 — 清洗对账报告 + 冻结清单
对照审计基线(行级 nR/nI) vs 清洗队列(培养级, I并入R), 登记定义差异
输出: E:\\ARMD\\audit_out\\reconcile_report.txt + data\\clean\\manifest.csv
"""
import os, hashlib
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
f = open(os.path.join(OUT, "reconcile_report.txt"), "w", encoding="utf-8")

def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n"); f.flush()

SITES = ["MGB", "Stanford", "UTSW"]
TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]
# 审计基线 (行级, 纯R / 纯I, 来自 step2/3/5)
AUDIT = {
 "MGB": {"meropenem":(1819,918),"ciprofloxacin":(3028,1714),"levofloxacin":(3668,1880),
         "ceftazidime":(1423,1078),"cefepime":(1022,958),"piperacillin_tazobactam":(1035,1494),
         "aztreonam":(538,1239),"amikacin":(300,326),"tobramycin":(807,298)},
 "Stanford": {"meropenem":(1144,435),"ciprofloxacin":(1904,1495),"levofloxacin":(1161,634),
         "ceftazidime":(1052,351),"cefepime":(1128,749),"piperacillin_tazobactam":(349,397),
         "aztreonam":(1211,702),"amikacin":(2149,680),"tobramycin":(1010,318)},
 "UTSW": {"meropenem":(1507,362),"ciprofloxacin":(3007,1660),"levofloxacin":(3751,1253),
         "ceftazidime":(2213,474),"cefepime":(2919,1281),"piperacillin_tazobactam":(499,388),
         "aztreonam":(2629,1295),"amikacin":(3255,1009),"tobramycin":(2163,351)},
}

p("=" * 100)
p("三站清洗对账报告 (2026-08-07)")
p("口径说明: 清洗队列=培养级(I并入R, 剔除prelim/阴性/无效标签, 精确药物匹配)")
p("审计基线=行级纯R/nI (step2/3/5); 差异来源: 行级→培养级去重 + I并入R + 剔除规则")
p("=" * 100)
rows = []
for site in SITES:
    for t in TASKS:
        fp = os.path.join(CLEAN, site, f"task_{t}.csv")
        df = pd.read_csv(fp, low_memory=False, encoding="utf-8-sig")
        n = len(df); nR = int(df["label"].sum()); rate = nR / n
        n_pat = df["anon_id"].nunique()
        audR, audI = AUDIT[site][t]
        rows.append(dict(site=site, task=t, n_cultures=n, n_patients=n_pat, nR_culture=nR,
                         R_rate=round(rate, 3), audit_row_R=audR, audit_row_I=audI,
                         sha256=hashlib.sha256(open(fp, "rb").read()).hexdigest()[:12]))
        p(f"{site:8s} {t:22s} 培养 {n:6,} 患者 {n_pat:5,} R率 {rate:.3f} "
          f"(R培养 {nR:,}) | 基线行级 R {audR:,} + I {audI:,}")

man = pd.DataFrame(rows)
man.to_csv(os.path.join(CLEAN, "manifest.csv"), index=False)
p("\n清单已存: data\\clean\\manifest.csv")

# 特征缺失率
p("\n特征完整性 (基座列缺失率):")
for site in SITES:
    base = pd.read_csv(os.path.join(CLEAN, site, "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
    miss = base[["age_bin","gender_bin","ward_h","adi","prior_pseudo_days","comorb_count"]].isna().mean()
    p(f"[{site}] 基座 {len(base):,} 培养 | 缺失率: " +
      " | ".join(f"{k}={v:.3f}" for k, v in miss.items()))
f.close()
print("完成", flush=True)
