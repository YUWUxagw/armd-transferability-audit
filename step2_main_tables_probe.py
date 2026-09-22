# -*- coding: utf-8 -*-
"""
ARMD 跨系统项目 — Step 2 第二优先审计
两侧微生物主表: 表头 + 取值抽样 + 命名词汇 + 标签分布 + MGB Q1 阳性事件数核查
输出: E:\ARMD\audit_out\Q1_MGB_nonfermenter_events.csv + report_step2.txt
"""
import os, time
import pandas as pd

MGB = r"E:\ARMD\ARMD-MGB\microbiology_cohort_deid_tj_updated.csv"
STA = r"E:\ARMD\ARMD-Stanford\microbiology_cultures_cohort.csv"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)

REPORT = os.path.join(OUT, "report_step2.txt")
f = open(REPORT, "w", encoding="utf-8")

pd.set_option("display.max_rows", 100)
pd.set_option("display.width", 220)
pd.set_option("display.max_colwidth", 60)

def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)
    f.write(f"[{time.strftime('%H:%M:%S')}] {msg}\n"); f.flush()

def p(*args):
    line = " ".join(str(a) for a in args)
    print(line, flush=True)
    f.write(line + "\n"); f.flush()

def kw_mask(s, kws):
    s2 = s.astype(str).str.lower()
    m = pd.Series(False, index=s.index)
    for k in kws:
        m |= s2.str.contains(k.lower(), na=False, regex=False)
    return m

# ============ MGB ============
log("MGB: 读表头")
head = pd.read_csv(MGB, nrows=3, encoding="utf-8-sig")
p(f"MGB 主表 {len(head.columns)} 列: {', '.join(head.columns)}")

log("MGB: 单次全扫描读关键列")
cols_mgb = ["organism","antibiotic","CLSI_2022_pheno","has_AST","mult_org_ast",
            "prelim_AST","neg_cx","order_proc_id_coded","order_time_jittered_utc_shifted"]
cols_mgb = [c for c in cols_mgb if c in head.columns]
mgb = pd.read_csv(MGB, usecols=cols_mgb, low_memory=False, encoding="utf-8-sig")
p(f"MGB 主表行数: {len(mgb):,}")

log("MGB: 标签与标志字段分布")
for c in ["CLSI_2022_pheno","has_AST","mult_org_ast","prelim_AST","neg_cx"]:
    if c in mgb.columns:
        p(f"\n[{c}]")
        p(mgb[c].value_counts(dropna=False).head(12).to_string())

log("MGB: 菌种命名 top20")
p("\n[organism] top20")
p(mgb["organism"].value_counts(dropna=False).head(20).to_string())

log("MGB: 抗生素词汇 top60")
p("\n[antibiotic] top60")
p(mgb["antibiotic"].value_counts(dropna=False).head(60).to_string())

log("MGB: 唯一性 & 时间")
p(f"唯一 order_proc_id: {mgb['order_proc_id_coded'].nunique():,} / 总行 {len(mgb):,}")
p(f"时间样例: {mgb['order_time_jittered_utc_shifted'].astype(str).head(5).tolist()}")
tt = pd.to_datetime(mgb["order_time_jittered_utc_shifted"], errors="coerce")
p(f"时间范围: {tt.min()} -> {tt.max()} | 解析失败: {int(tt.isna().sum())}")

log("MGB: 目标菌种命名匹配")
pseudo = kw_mask(mgb["organism"], ["pseudomonas","aerug"])
acine  = kw_mask(mgb["organism"], ["acinetobacter","baumannii"])
p(f"\nPseudomonas 类: {int(pseudo.sum()):,} 行")
p(mgb.loc[pseudo,"organism"].value_counts().head(12).to_string())
p(f"\nAcinetobacter 类: {int(acine.sum()):,} 行")
p(mgb.loc[acine,"organism"].value_counts().head(12).to_string())

log("MGB: Q1 阳性事件数 (非发酵菌 + 大肠杆菌参照)")
TASKS = {
  "P_aeruginosa": ["ceftazidime","CAZ","cefepime","FEP","piperacillin","tazobactam",
                   "meropenem","MEM","ciprofloxacin","CIP","levofloxacin","amikacin",
                   "tobramycin","aztreonam"],
  "A_baumannii":  ["meropenem","MEM","imipenem","IPM","sulbactam","ciprofloxacin","CIP",
                   "levofloxacin","minocycline","tigecycline"],
}
REF  = {"E_coli": ["ceftriaxone","CRO","ciprofloxacin","CIP"]}
ORG_KW = {"P_aeruginosa": ["pseudomonas","aerug"],
          "A_baumannii":  ["acinetobacter","baumannii"],
          "E_coli":       ["escherichia","e coli","ecoli","coli"]}
pheno = mgb["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
rows = []
for org, abxlist in {**TASKS, **REF}.items():
    om = kw_mask(mgb["organism"], ORG_KW[org])
    for abx in abxlist:
        idx = om & kw_mask(mgb["antibiotic"], [abx])
        n = int(idx.sum())
        if n == 0:
            continue
        pv = pheno[idx]
        nR = int(pv.isin(["R","RESISTANT"]).sum())
        nS = int(pv.isin(["S","SUSCEPTIBLE"]).sum())
        nI = int(pv.isin(["I","INTERMEDIATE"]).sum())
        d = nR + nS + nI
        rows.append(dict(organism=org, antibiotic=abx, n_total=n, nR=nR, nS=nS, nI=nI,
                         R_rate=round(nR/d, 3) if d else None))
q1 = pd.DataFrame(rows).sort_values(["organism","nR"], ascending=[True,False])
q1.to_csv(os.path.join(OUT, "Q1_MGB_nonfermenter_events.csv"), index=False)
p("\n===== Q1_MGB_nonfermenter_events.csv =====")
p(q1.to_string(index=False))

# ============ Stanford ============
log("Stanford: 读表头")
sta_head = pd.read_csv(STA, nrows=3, encoding="utf-8-sig")
p(f"\nStanford 主表 {len(sta_head.columns)} 列: {', '.join(sta_head.columns)}")
p(sta_head.head(3).to_string())

log("Stanford: 单次扫描读关键列")
use = [c for c in ["organism","antibiotic","susceptibility","was_positive",
                   "culture_description","order_proc_id_coded","order_time_jittered_utc"]
       if c in sta_head.columns]
sta = pd.read_csv(STA, usecols=use, low_memory=False, encoding="utf-8-sig")
p(f"Stanford 主表行数: {len(sta):,}")
for c in ["was_positive","susceptibility","culture_description"]:
    if c in sta.columns:
        p(f"\n[{c}]")
        p(sta[c].value_counts(dropna=False).head(15).to_string())
if "organism" in sta.columns:
    p("\n[organism] top20")
    p(sta["organism"].value_counts(dropna=False).head(20).to_string())
if "antibiotic" in sta.columns:
    p("\n[antibiotic] top50")
    p(sta["antibiotic"].value_counts(dropna=False).head(50).to_string())
if "order_proc_id_coded" in sta.columns:
    p(f"\n唯一 order_proc_id: {sta['order_proc_id_coded'].nunique():,} / 总行 {len(sta):,}")
if "order_time_jittered_utc" in sta.columns:
    p(f"时间样例: {sta['order_time_jittered_utc'].astype(str).head(5).tolist()}")
    tts = pd.to_datetime(sta["order_time_jittered_utc"], errors="coerce")
    p(f"时间范围: {tts.min()} -> {tts.max()} | 解析失败: {int(tts.isna().sum())}")

log("Stanford: 目标菌种体量")
if "organism" in sta.columns:
    sp = kw_mask(sta["organism"], ["pseudomonas","aerug"])
    sa = kw_mask(sta["organism"], ["acinetobacter","baumannii"])
    p(f"\nPseudomonas 类: {int(sp.sum()):,} 行")
    p(sta.loc[sp,"organism"].value_counts().head(10).to_string())
    p(f"\nAcinetobacter 类: {int(sa.sum()):,} 行")
    p(sta.loc[sa,"organism"].value_counts().head(10).to_string())

log("完成")
p(f"\n报告与 CSV 输出目录: {OUT}")
f.close()
