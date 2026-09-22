# -*- coding: utf-8 -*-
"""
ARMD Step 5 — UTSW 三站点可行性探测
PA 事件数 + 结构 + 时间 + 标签形态
输出: E:\ARMD\audit_out\Q1_UTSW_PA_events.csv + report_step5_utsw.txt
"""
import os, time
import pandas as pd

UTSW = r"E:\ARMD\ARMD-UTSW\microbiology_cultures_cohort.csv"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report_step5_utsw.txt")
f = open(REPORT, "w", encoding="utf-8")

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 220)

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); f.write(line + "\n"); f.flush()

def p(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True); f.write(line + "\n"); f.flush()

def norm(s):
    return (s.astype(str).str.lower()
             .str.replace("/", "_", regex=False)
             .str.replace("-", "_", regex=False)
             .str.strip())

DRUGS_PA = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
            "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]

log("UTSW: 表头")
head = pd.read_csv(UTSW, nrows=3, encoding="utf-8-sig")
p(f"UTSW 主表 {len(head.columns)} 列: {', '.join(head.columns)}")
tcol = [c for c in head.columns if "time" in c.lower()]
tcol = tcol[0] if tcol else "order_time_jittered"
p(f"时间列: {tcol}")

use = [c for c in ["organism","antibiotic","susceptibility","was_positive",
                   "order_proc_id_coded", tcol] if c in head.columns]
log("UTSW: 单次全扫描")
sta = pd.read_csv(UTSW, usecols=use, low_memory=False, encoding="utf-8-sig")
p(f"UTSW 主表行数: {len(sta):,}")

pos = sta["was_positive"].astype(str).str.strip().isin(["1","1.0"])
org = sta["organism"].astype(str)
pa_mask = org.str.contains("PSEUDOMONAS AERUGINOSA", na=False)
pa_cf = pa_mask & org.str.contains("CF", na=False)
p(f"阳性培养行: {int(pos.sum()):,}")
p(f"PA 行(全部): {int(pa_mask.sum()):,} | CF 标注: {int(pa_cf.sum()):,} | 非 CF: {int((pa_mask & ~pa_cf).sum()):,}")
p(f"PA 唯一培养: {int(sta.loc[pa_mask,'order_proc_id_coded'].nunique()):,}")
p("\nPA 命名:")
p(org[pa_mask].value_counts().head(12).to_string())

p("\n[susceptibility 全量分布]")
p(sta["susceptibility"].value_counts(dropna=False).to_string())

if tcol in sta.columns:
    tt = pd.to_datetime(sta[tcol], errors="coerce")
    p(f"\n时间范围: {tt.min()} -> {tt.max()} | 解析失败: {int(tt.isna().sum())}")

anorm = norm(sta["antibiotic"])
sus = sta["susceptibility"].astype(str).str.upper().str.strip()
isR = sus.isin(["RESISTANT"]); isS = sus.isin(["SUSCEPTIBLE"]); isI = sus.isin(["INTERMEDIATE"])
pa_pos = pos & pa_mask

p("\n===== UTSW PA 药物事件数 (阳性培养) =====")
rows = []
for d in DRUGS_PA:
    idx = pa_pos & (anorm == d)
    n = int(idx.sum())
    nR = int(isR[idx].sum()); nS = int(isS[idx].sum()); nI = int(isI[idx].sum())
    d0 = nR + nS + nI
    rows.append(dict(drug=d, n_tested=n, nR=nR, nS=nS, nI=nI, n_invalid=n-nR-nS-nI,
                     R_rate=round(nR/d0,3) if d0 else None))
q = pd.DataFrame(rows).sort_values("nR", ascending=False)
p(q.to_string(index=False))
q.to_csv(os.path.join(OUT, "Q1_UTSW_PA_events.csv"), index=False)

log("完成")
p(f"\n输出目录: {OUT}")
f.close()
