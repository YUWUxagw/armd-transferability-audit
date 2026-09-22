# -*- coding: utf-8 -*-
"""
ARMD Step 3 — Stanford 侧事件数核查 (方案 A 决策数据)
PA/AB 药物事件数 + susceptibility 全分布 + DTR 面板覆盖率 + implied_susceptibility 结构
输出: E:\ARMD\audit_out\Q1_Stanford_*.csv + report_step3_stanford.txt
"""
import os, time
import pandas as pd

STA   = r"E:\ARMD\ARMD-Stanford\microbiology_cultures_cohort.csv"
IMPL  = r"E:\ARMD\ARMD-Stanford\microbiology_cultures_implied_susceptibility.csv"
RULES = r"E:\ARMD\ARMD-Stanford\implied_susceptibility_rules.csv"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report_step3_stanford.txt")
f = open(REPORT, "w", encoding="utf-8")

pd.set_option("display.max_rows", 200)
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
DRUGS_AB = ["meropenem","ciprofloxacin","levofloxacin","ampicillin_sulbactam",
            "imipenem","minocycline","tigecycline"]
DTR_SET  = {"ceftazidime","cefepime","piperacillin_tazobactam","meropenem",
            "ciprofloxacin","levofloxacin"}

log("Stanford: 扫描主表关键列")
use = ["order_proc_id_coded","organism","antibiotic","susceptibility","was_positive"]
sta = pd.read_csv(STA, usecols=use, low_memory=False, encoding="utf-8-sig")
p(f"Stanford 主表行数: {len(sta):,}")

pos = sta["was_positive"].astype(str).str.strip().isin(["1","1.0"])
org = sta["organism"].astype(str)
pa_mask = org.str.contains("PSEUDOMONAS AERUGINOSA", na=False)
pa_cf   = pa_mask & org.str.contains("CF", na=False)
ab_mask = org.str.contains("ACINETOBACTER", na=False)
ab_baum = ab_mask & org.str.contains("BAUMANNII", na=False)

p(f"阳性培养行: {int(pos.sum()):,}")
p(f"PA 行(全部): {int(pa_mask.sum()):,} | 其中 CF 标注: {int(pa_cf.sum()):,} | 非 CF: {int((pa_mask & ~pa_cf).sum()):,}")
p(f"PA 唯一培养: {int(sta.loc[pa_mask, 'order_proc_id_coded'].nunique()):,}")
p(f"AB 行(全部): {int(ab_mask.sum()):,} | 其中 BAUMANNII: {int(ab_baum.sum()):,} | AB 唯一培养: {int(sta.loc[ab_mask, 'order_proc_id_coded'].nunique()):,}")

p("\n[susceptibility 全量分布]")
p(sta["susceptibility"].value_counts(dropna=False).to_string())

anorm = norm(sta["antibiotic"])
sus = sta["susceptibility"].astype(str).str.upper().str.strip()
isR = sus.isin(["RESISTANT"]); isS = sus.isin(["SUSCEPTIBLE"]); isI = sus.isin(["INTERMEDIATE"])

def count_events(rows, drugs, label):
    p(f"\n===== {label}: 药物事件数 =====")
    out = []
    for d in drugs:
        idx = rows & (anorm == d)
        n = int(idx.sum())
        nR = int(isR[idx].sum()); nS = int(isS[idx].sum()); nI = int(isI[idx].sum())
        n_invalid = n - nR - nS - nI
        d0 = nR + nS + nI
        out.append(dict(drug=d, n_tested=n, nR=nR, nS=nS, nI=nI, n_invalid=n_invalid,
                        R_rate=round(nR/d0, 3) if d0 else None))
    df = pd.DataFrame(out).sort_values("nR", ascending=False)
    p(df.to_string(index=False))
    return df

pa_pos  = pos & pa_mask
ab_pos  = pos & ab_mask
q_pa      = count_events(pa_pos, DRUGS_PA, "PA (含 CF, 阳性培养)")
q_pa_nocf = count_events(pa_pos & ~pa_cf, DRUGS_PA, "PA 排除 CF 标注 (阳性培养)")
q_ab      = count_events(ab_pos, DRUGS_AB, "AB (阳性培养)")
q_ab_baum = count_events(pos & ab_baum, DRUGS_AB, "AB 仅 BAUMANNII (阳性培养)")

log("DTR 面板覆盖率 (PA, 阳性培养, 培养级)")
pa_cx = sta.loc[pa_pos, ["order_proc_id_coded","antibiotic"]].copy()
pa_cx["drug"] = anorm[pa_pos]
pa_cx = pa_cx[pa_cx["drug"].isin(DTR_SET)]
grp = pa_cx.groupby("order_proc_id_coded")["drug"].agg(lambda x: set(x))
n_cx = len(grp)
p(f"PA 阳性培养(含至少一个 DTR 药检测): {n_cx:,}")
for k in [6,5,4,3]:
    cnt = int((grp.map(len) >= k).sum())
    p(f"  覆盖 >= {k}/6 个 DTR 药的培养: {cnt:,} ({100*cnt/max(n_cx,1):.1f}%)")
full6 = grp[grp.map(len) >= 6]
p(f"  6/6 全测培养数: {len(full6):,}")

log("implied_susceptibility 结构探查")
try:
    imp = pd.read_csv(IMPL, nrows=5, encoding="utf-8-sig")
    p(f"implied_susceptibility 列: {list(imp.columns)}")
    p(imp.head(5).to_string())
except Exception as e:
    p(f"读取失败: {e}")

log("implied_susceptibility_rules 读取")
try:
    rules = pd.read_csv(RULES, encoding="utf-8-sig")
    p(f"rules 行数: {len(rules):,} | 列: {list(rules.columns)}")
    if "Organism" in rules.columns:
        p("\n规则涉及菌种 top20:")
        p(rules["Organism"].value_counts().head(20).to_string())
    if "Antibiotic" in rules.columns:
        p("\n规则涉及抗生素 top30:")
        p(rules["Antibiotic"].value_counts().head(30).to_string())
    p("\n规则样例 head15:")
    p(rules.head(15).to_string())
except Exception as e:
    p(f"读取失败: {e}")

q_pa.to_csv(os.path.join(OUT, "Q1_Stanford_PA_events.csv"), index=False)
q_pa_nocf.to_csv(os.path.join(OUT, "Q1_Stanford_PA_noCF_events.csv"), index=False)
q_ab.to_csv(os.path.join(OUT, "Q1_Stanford_AB_events.csv"), index=False)
q_ab_baum.to_csv(os.path.join(OUT, "Q1_Stanford_AB_baumannii_events.csv"), index=False)
log("完成")
p(f"\n输出目录: {OUT}")
f.close()
