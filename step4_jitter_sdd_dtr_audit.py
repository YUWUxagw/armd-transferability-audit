# -*- coding: utf-8 -*-
"""
ARMD Step 4 — 批次2: jitter 相对时间保真验证 + PA SDD 计数 + DTR-PA 事件数 (双侧)
1) prior_micro / prior_org / prior_abx 预计算相对时间字段与平移绝对时间的一致性检查
2) 主表患者内连续培养间隔分布 (MGB vs Stanford 对照)
3) MGB PA 各药 SDD 计数
4) DTR-PA 事件数 (6/6 与 >=3/6 面板, 严格 R 与 R-or-I 两种定义)
输出: E:\\ARMD\\audit_out\\report_step4.txt
"""
import os, time
import numpy as np
import pandas as pd

MGB = r"E:\ARMD\ARMD-MGB\microbiology_cohort_deid_tj_updated.csv"
STA = r"E:\ARMD\ARMD-Stanford\microbiology_cultures_cohort.csv"
P_MICRO = r"E:\ARMD\ARMD-MGB\prior_micro_deid_tj.csv"
P_ABX   = r"E:\ARMD\ARMD-MGB\prior_abx_deid_tj.csv"
P_ORG   = r"E:\ARMD\ARMD-MGB\prior_org_deid_tj.csv"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report_step4.txt")
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

DTR_SET = {"ceftazidime","cefepime","piperacillin_tazobactam","meropenem",
           "ciprofloxacin","levofloxacin"}

# ============ 1. 预计算相对时间一致性检查 (抽样) ============
log("1. 预计算相对时间字段一致性 (prior_micro / prior_org / prior_abx 抽样)")
def rel_check(path, nrows, t_col, rel_col, evt_col, label):
    try:
        df = pd.read_csv(path, nrows=nrows, usecols=[t_col, rel_col, evt_col], encoding="utf-8-sig")
    except Exception as e:
        p(f"[{label}] 读取失败: {e}"); return
    t1 = pd.to_datetime(df[t_col], errors="coerce")   # 当前培养时间(平移后)
    t2 = pd.to_datetime(df[evt_col], errors="coerce") # 既往事件时间(平移后)
    rel = pd.to_numeric(df[rel_col], errors="coerce") # 预计算相对天数
    d = (t1 - t2).dt.days
    ok = t1.notna() & t2.notna() & rel.notna()
    d, rel = d[ok], rel[ok]
    n = len(d)
    if n == 0:
        p(f"[{label}] 有效配对 0"); return
    err_pos = (d - rel).abs(); err_neg = (d + rel).abs()
    match_pos = float((err_pos <= 1).mean()); match_neg = float((err_neg <= 1).mean())
    p(f"[{label}] n={n:,} | 相对列样例: {df[rel_col].astype(str).head(3).tolist()}")
    p(f"   约定A(rel==间隔天数): {match_pos:.3f} | 约定B(rel==负间隔): {match_neg:.3f} | "
      f"间隔-相对差值P99: {int(np.quantile(err_pos, .99))}天")

rel_check(P_MICRO, 500000, "order_time_jittered_utc_shifted", "prior_AST_time_to_culture", "prior_AST_DTS_shifted", "prior_micro")
rel_check(P_ABX,   300000, "order_time_jittered_utc_shifted", "last_dose_to_culture",      "last_dose_DT_shifted", "prior_abx")
rel_check(P_ORG,   200000, "order_time_jittered_utc_shifted", "prior_org_days_to_culture", "prior_org_recorded_time_shifted", "prior_org")

# ============ 2. 患者内连续培养间隔分布 ============
log("2. 患者内连续培养间隔分布 (MGB vs Stanford)")
def gap_stats(df, label, cap_patients=3000):
    t = pd.to_datetime(df["order_time_jittered_utc_shifted"].astype(str), errors="coerce")
    cult = df[["anon_id","order_proc_id_coded"]].assign(t=t).dropna()
    cult = cult.drop_duplicates(["anon_id","order_proc_id_coded"])
    sizes = cult.groupby("anon_id")["t"].size()
    multi = sizes[sizes >= 5]
    if len(multi) > cap_patients:
        keep = multi.sample(cap_patients, random_state=42).index
    else:
        keep = multi.index
    gaps = []
    for aid, g in cult[cult["anon_id"].isin(keep)].groupby("anon_id"):
        ts = g["t"].sort_values().values
        gaps.extend(np.diff(ts.astype("datetime64[D]").astype(int)))
    gaps = np.array(gaps)
    n_neg = float((gaps < 0).mean())
    n_zero = float((gaps == 0).mean())
    p(f"[{label}] 患者数(>=5培养): {len(keep):,} | 间隔数: {len(gaps):,}")
    p(f"   负间隔: {n_neg:.4f} | 零间隔: {n_zero:.4f} | 中位: {np.median(gaps):.0f}天 | "
      f"P95: {np.quantile(gaps,.95):.0f}天 | >1年: {float((gaps>365).mean()):.3f} | >5年: {float((gaps>1825).mean()):.4f} | 最大: {int(gaps.max())}天")

cols_mgb = ["anon_id","order_proc_id_coded","order_time_jittered_utc_shifted"]
mgb_t = pd.read_csv(MGB, usecols=cols_mgb, low_memory=False, encoding="utf-8-sig")
gap_stats(mgb_t, "MGB")
sta_t = pd.read_csv(STA, usecols=["anon_id","order_proc_id_coded","order_time_jittered_utc"], low_memory=False, encoding="utf-8-sig")
sta_t = sta_t.rename(columns={"order_time_jittered_utc":"order_time_jittered_utc_shifted"})
gap_stats(sta_t, "Stanford")

# ============ 3. MGB PA SDD 计数 ============
log("3. MGB PA 各药 SDD 计数 (全表)")
cols = ["organism","antibiotic","CLSI_2022_pheno","order_proc_id_coded"]
mgb2 = pd.read_csv(MGB, usecols=cols, low_memory=False, encoding="utf-8-sig")
pa = mgb2[mgb2["organism"].astype(str).str.contains("PSEUDOMONAS AERUGINOSA", na=False)]
pheno = pa["CLSI_2022_pheno"].astype(str)
sdd = pheno.str.upper().str.contains("DOSE DEPENDENT", na=False)
p(f"PA 行数: {len(pa):,} | SDD 行数: {int(sdd.sum()):,}")
p("SDD 按药物分布 (top15):")
p(pa.loc[sdd, "antibiotic"].value_counts().head(15).to_string())
p("\nPA 非常规表型值 (非 S/R/I/SDD):")
other = pa[~pheno.str.upper().str.strip().isin(["SUSCEPTIBLE","RESISTANT","INTERMEDIATE"]) & ~sdd]
p(other["CLSI_2022_pheno"].value_counts(dropna=False).head(10).to_string())

# ============ 4. DTR-PA 事件数 (双侧) ============
log("4. DTR-PA 事件数")
def dtr_counts(df, pheno_col, label):
    pa = df[df["organism"].astype(str).str.contains("PSEUDOMONAS AERUGINOSA", na=False)]
    ph = pa[pheno_col].astype(str).str.upper().str.strip()
    R = ph.eq("RESISTANT")
    NS = ph.isin(["RESISTANT","INTERMEDIATE"])
    d = pa.assign(drug=norm(pa["antibiotic"]), R=R, NS=NS)
    d = d[d["drug"].isin(DTR_SET)]
    g = d.groupby("order_proc_id_coded").agg(
        tested=("drug","nunique"),
        allR=("R","all"), allNS=("NS","all"))
    p(f"\n[{label}] PA 培养(测过>=1个DTR药): {len(g):,}")
    for k in [6,5,4,3]:
        sub = g[g["tested"]>=k]
        p(f"  覆盖>={k}/6: {len(sub):,} | 其中全部R: {int(sub['allR'].sum()):,} | 全部R或I: {int(sub['allNS'].sum()):,}")

dtr_counts(mgb2, "CLSI_2022_pheno", "MGB")
sta2 = pd.read_csv(STA, usecols=["organism","antibiotic","susceptibility","order_proc_id_coded","was_positive"], low_memory=False, encoding="utf-8-sig")
sta2 = sta2[sta2["was_positive"].astype(str).str.strip().isin(["1","1.0"])]
dtr_counts(sta2, "susceptibility", "Stanford")

log("完成")
p(f"\n输出目录: {OUT}")
f.close()
