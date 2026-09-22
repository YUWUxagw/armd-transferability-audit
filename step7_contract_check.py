# -*- coding: utf-8 -*-
"""
ARMD 数据契约检查 (构建前第一闸)
A. 断言: 列存在 / 键完整性 / 标签取值集 / 相对时间非负(三站全量)
B. 词表全量提取 → E:\\ARMD\\project\\vocab\\*.csv (字典生成输入)
C. 遗留开放项: UTSW 耐药史时间分布 / UTSW DTR / Stanford tobramycin 无效行 / MGB prior_micro NaN 占位
输出: E:\\ARMD\\audit_out\\report_step7_contract.txt
"""
import os, time
import pandas as pd
from collections import Counter

M = r"E:\ARMD\ARMD-MGB"; S = r"E:\ARMD\ARMD-Stanford"; U = r"E:\ARMD\ARMD-UTSW"
VOCAB = r"E:\ARMD\project\vocab"; OUT = r"E:\ARMD\audit_out"
os.makedirs(VOCAB, exist_ok=True); os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "report_step7_contract.txt")
f = open(REPORT, "w", encoding="utf-8")

fails = []
def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); f.write(line + "\n"); f.flush()
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n"); f.flush()
def fail(tag, msg):
    fails.append(tag); p(f"[FAIL] {tag}: {msg}")

def norm(s):
    return (s.astype(str).str.lower().str.replace("/", "_", regex=False)
             .str.replace("-", "_", regex=False).str.strip())

# ============ A1 列存在断言 (契约列) ============
log("A1. 列存在断言")
A1 = {
 M: {"microbiology_cohort_deid_tj_updated.csv": ["anon_id","order_proc_id_coded","organism","antibiotic","CLSI_2022_pheno","has_AST","neg_cx","prelim_AST","mult_org_ast","culture_description"],
     "demographics_deid_tj.csv": ["anon_id","age","gender"],
     "ward_type_deid_tj.csv": ["order_proc_id_coded","hosp_ward_IP","hosp_ward_OP","hosp_ward_ER","hosp_ward_UC","hosp_ward_day_surg"],
     "prior_abx_deid_tj.csv": ["anon_id","drug_class","last_dose_to_culture","medication_name","drug_code"],
     "prior_org_deid_tj.csv": ["anon_id","prior_org","prior_org_days_to_culture"],
     "prior_micro_deid_tj.csv": ["anon_id","organism","antibiotic","CLSI_2022_pheno","prior_AST_time_to_culture"],
     "comorbidity_deid_tj.csv": ["anon_id","ICD10","category"],
     "nursing_home_visits_deid_tj.csv": ["anon_id","nursing_home_visit_culture"]},
 S: {"microbiology_cultures_cohort.csv": ["anon_id","order_proc_id_coded","organism","antibiotic","susceptibility","was_positive","culture_description"],
     "microbiology_cultures_ward_info.csv": ["order_proc_id_coded","hosp_ward_IP","hosp_ward_OP","hosp_ward_ER","hosp_ward_ICU"],
     "microbiology_cultures_prior_med.csv": ["anon_id","medication_name","medication_time_to_culturetime","medication_category"],
     "microbiology_cultures_antibiotic_class_exposure.csv": ["anon_id","antibiotic_class","time_to_culturetime"],
     "microbiology_cultures_antibiotic_subtype_exposure.csv": ["anon_id","antibiotic_subtype","medication_time_to_cultureTime"],
     "microbiology_cultures_comorbidity.csv": ["anon_id","comorbidity_component","comorbidity_component_start_days_culture"],
     "microbiology_culture_prior_infecting_organism.csv": ["anon_id","prior_organism","prior_infecting_organism_days_to_culutre"],
     "microbiology_cultures_priorprocedures.csv": ["anon_id","procedure_description","procedure_time_to_culturetime"],
     "microbiology_cultures_microbial_resistance.csv": ["anon_id","organism","antibiotic","resistant_time_to_culturetime"],
     "microbiology_cultures_nursing_home_visits.csv": ["anon_id","nursing_home_visit_culture"],
     "microbiology_cultures_demographics.csv": ["anon_id","age","gender"]},
 U: {"microbiology_cultures_cohort.csv": ["anon_id","order_proc_id_coded","organism","antibiotic","susceptibility","was_positive","culture_description"],
     "microbiology_cultures_ward_info.csv": ["order_proc_id_coded","hosp_ward_IP","hosp_ward_OP","hosp_ward_ER","hosp_ward_ICU"],
     "microbiology_cultures_prior_med.csv": ["anon_id","medication_name","medication_time_to_culturetime","medication_category"],
     "microbiology_cultures_antibiotic_class_exposure.csv": ["anon_id","antibiotic_class","time_to_culturetime"],
     "microbiology_cultures_antibiotic_subtype_exposure.csv": ["anon_id","antibiotic_subtype","medication_time_to_culturetime"],
     "microbiology_cultures_comorbidity.csv": ["anon_id","comorbidity_component","comorbidity_component_start_days_culture"],
     "microbiology_cultures_prior_infecting_organism.csv": ["anon_id","prior_organism","prior_infecting_organism_days_to_culture"],
     "microbiology_cultures_prior_procedures.csv": ["anon_id","procedure_description","procedure_time_to_culturetime"],
     "microbiology_cultures_microbial_resistance.csv": ["anon_id","organism","antibiotic","resistant_time_to_culturetime"],
     "microbiology_cultures_nursing_home_visits.csv": ["anon_id","nursing_home_visit_culture"],
     "microbiology_cultures_demographics.csv": ["anon_id","age","gender"]},
}
for site, tbls in A1.items():
    for fn, cols in tbls.items():
        try:
            have = set(pd.read_csv(os.path.join(site, fn), nrows=0).columns)
            miss = [c for c in cols if c not in have]
            if miss: fail(f"A1 {os.path.basename(site)}/{fn}", f"缺列 {miss}")
        except Exception as e:
            fail(f"A1 {os.path.basename(site)}/{fn}", str(e))
p(f"A1 完成: 断言表 {sum(len(v) for v in A1.values())} 张")

# ============ A2 标签取值集 ============
log("A2. 标签取值集")
def label_check(path, col, allowed, tag):
    bad = Counter()
    for chunk in pd.read_csv(path, usecols=[col], chunksize=2000000, low_memory=False, encoding="utf-8-sig"):
        for v, c in chunk[col].astype(str).value_counts().items():
            if v not in allowed: bad[v] += c
    if bad: fail("A2 " + tag, f"意外取值 {dict(bad)}")
    else: p(f"[A2 {tag}] 取值集合规")

label_check(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"), "CLSI_2022_pheno",
            {"Susceptible","Resistant","Intermediate","Susceptible dose dependent","Susceptible dose-dependent","Non-susceptible","nan"}, "MGB CLSI_2022_pheno")
label_check(os.path.join(S, "microbiology_cultures_cohort.csv"), "susceptibility",
            {"Susceptible","Resistant","Intermediate","Inconclusive","Synergism","Null"}, "Stanford susceptibility")
label_check(os.path.join(U, "microbiology_cultures_cohort.csv"), "susceptibility",
            {"Susceptible","Resistant","Intermediate","Inconclusive","Null"}, "UTSW susceptibility")

# ============ A3 相对时间非负 (三站全量) ============
log("A3. 相对时间非负扫描")
A3 = [
 (M, "prior_abx_deid_tj.csv", "last_dose_to_culture", "MGB prior_abx"),
 (M, "prior_org_deid_tj.csv", "prior_org_days_to_culture", "MGB prior_org"),
 (M, "prior_micro_deid_tj.csv", "prior_AST_time_to_culture", "MGB prior_micro"),
 (M, "prior_procedures_deid_tj.csv", "procedure_days_culture", "MGB procedures"),
 (M, "nursing_home_visits_deid_tj.csv", "nursing_home_visit_culture", "MGB NH"),
 (S, "microbiology_cultures_prior_med.csv", "medication_time_to_culturetime", "S prior_med"),
 (S, "microbiology_cultures_antibiotic_class_exposure.csv", "time_to_culturetime", "S class_exp"),
 (S, "microbiology_cultures_antibiotic_subtype_exposure.csv", "medication_time_to_cultureTime", "S subtype_exp"),
 (S, "microbiology_cultures_comorbidity.csv", "comorbidity_component_start_days_culture", "S comorb_start"),
 (S, "microbiology_cultures_comorbidity.csv", "comorbidity_component_end_days_culture", "S comorb_end"),
 (S, "microbiology_culture_prior_infecting_organism.csv", "prior_infecting_organism_days_to_culutre", "S prior_org"),
 (S, "microbiology_cultures_priorprocedures.csv", "procedure_time_to_culturetime", "S procedures"),
 (S, "microbiology_cultures_microbial_resistance.csv", "resistant_time_to_culturetime", "S mic_res"),
 (U, "microbiology_cultures_prior_med.csv", "medication_time_to_culturetime", "U prior_med"),
 (U, "microbiology_cultures_antibiotic_class_exposure.csv", "time_to_culturetime", "U class_exp"),
 (U, "microbiology_cultures_antibiotic_subtype_exposure.csv", "medication_time_to_culturetime", "U subtype_exp"),
 (U, "microbiology_cultures_comorbidity.csv", "comorbidity_component_start_days_culture", "U comorb_start"),
 (U, "microbiology_cultures_prior_infecting_organism.csv", "prior_infecting_organism_days_to_culture", "U prior_org"),
 (U, "microbiology_cultures_prior_procedures.csv", "procedure_time_to_culturetime", "U procedures"),
 (U, "microbiology_cultures_microbial_resistance.csv", "resistant_time_to_culturetime", "U mic_res"),
]
for site, fn, col, tag in A3:
    mn = None; n = 0; neg = 0
    for chunk in pd.read_csv(os.path.join(site, fn), usecols=[col], chunksize=2000000, low_memory=False, encoding="utf-8-sig"):
        v = pd.to_numeric(chunk[col], errors="coerce")
        n += int(v.notna().sum())
        neg += int((v < 0).sum())
        m = v.min()
        if pd.notna(m) and (mn is None or m < mn): mn = m
    status = "OK" if (neg == 0) else "VIOLATION"
    if neg: fail("A3 " + tag, f"负值 {neg}/{n}")
    p(f"[A3 {tag}] n={n:,} min={mn} 负值={neg} -> {status}")

# ============ B 词表全量提取 ============
log("B. 词表全量提取")
def dump_vocab(path, col, tag, limit=None):
    c = Counter()
    for chunk in pd.read_csv(path, usecols=[col], chunksize=2000000, low_memory=False, encoding="utf-8-sig"):
        for v, k in chunk[col].astype(str).value_counts().items():
            c[v] += k
    df = pd.DataFrame(sorted(c.items(), key=lambda x: -x[1]), columns=["value", "count"])
    if limit: df = df.head(limit)
    fp = os.path.join(VOCAB, f"vocab_{tag}.csv")
    df.to_csv(fp, index=False)
    p(f"[vocab_{tag}] {len(df):,} 个取值 -> {os.path.basename(fp)}")
    return df

dump_vocab(os.path.join(M, "prior_abx_deid_tj.csv"), "drug_class", "MGB_drug_class")
dump_vocab(os.path.join(M, "prior_abx_deid_tj.csv"), "medication_name", "MGB_drug_names")
dump_vocab(os.path.join(M, "prior_abx_deid_tj.csv"), "drug_code", "MGB_drug_codes")
dump_vocab(os.path.join(M, "prior_org_deid_tj.csv"), "prior_org", "MGB_prior_org")
dump_vocab(os.path.join(M, "comorbidity_deid_tj.csv"), "category", "MGB_comorbidity_category")
dump_vocab(os.path.join(M, "demographics_deid_tj.csv"), "age", "MGB_age_bins")
dump_vocab(os.path.join(S, "microbiology_cultures_demographics.csv"), "age", "S_age_bins")
dump_vocab(os.path.join(U, "microbiology_cultures_demographics.csv"), "age", "U_age_bins")
dump_vocab(os.path.join(S, "microbiology_cultures_antibiotic_subtype_exposure.csv"), "antibiotic_subtype", "S_antibiotic_subtype")
dump_vocab(os.path.join(U, "microbiology_cultures_antibiotic_subtype_exposure.csv"), "antibiotic_subtype", "U_antibiotic_subtype")
dump_vocab(os.path.join(S, "microbiology_cultures_prior_med.csv"), "medication_category", "S_medication_category")
dump_vocab(os.path.join(U, "microbiology_cultures_prior_med.csv"), "medication_category", "U_medication_category")
dump_vocab(os.path.join(S, "microbiology_cultures_comorbidity.csv"), "comorbidity_component", "S_comorbidity_component")
dump_vocab(os.path.join(U, "microbiology_cultures_comorbidity.csv"), "comorbidity_component", "U_comorbidity_component")
dump_vocab(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"), "culture_description", "MGB_specimen")
dump_vocab(os.path.join(S, "microbiology_cultures_cohort.csv"), "culture_description", "S_specimen")
dump_vocab(os.path.join(U, "microbiology_cultures_cohort.csv"), "culture_description", "U_specimen")

# ============ C 开放项 ============
log("C. 开放项")
# C1 UTSW microbial_resistance 时间分布
v = pd.to_numeric(pd.read_csv(os.path.join(U, "microbiology_cultures_microbial_resistance.csv"),
    usecols=["resistant_time_to_culturetime"], nrows=500000, low_memory=False, encoding="utf-8-sig")["resistant_time_to_culturetime"], errors="coerce").dropna()
p(f"[C1 UTSW mic_res] n={len(v):,} min={v.min()} 负值={float((v<0).mean()):.3f} 零值={float((v==0).mean()):.3f} 中位={v.median():.0f} P75={v.quantile(.75):.0f} max={v.max()}")

# C2 UTSW DTR 面板
cols = ["order_proc_id_coded","organism","antibiotic","susceptibility","was_positive"]
st = pd.read_csv(os.path.join(U, "microbiology_cultures_cohort.csv"), usecols=cols, low_memory=False, encoding="utf-8-sig")
pos = st["was_positive"].astype(str).str.strip().isin(["1","1.0"])
pa = st["organism"].astype(str).str.contains("PSEUDOMONAS AERUGINOSA", na=False) & pos
dtr = set(["ceftazidime","cefepime","piperacillin_tazobactam","meropenem","ciprofloxacin","levofloxacin"])
sus = st["susceptibility"].astype(str).str.upper().str.strip()
R = sus.eq("RESISTANT"); NS = sus.isin(["RESISTANT","INTERMEDIATE"])
dd = st[pa].assign(drug=norm(st.loc[pa, "antibiotic"]), R=R[pa], NS=NS[pa])
dd = dd[dd["drug"].isin(dtr)]
g = dd.groupby("order_proc_id_coded").agg(tested=("drug","nunique"), allR=("R","all"), allNS=("NS","all"))
p(f"[C2 UTSW DTR] PA 培养(≥1 DTR药) {len(g):,} | ≥6/6: {int((g['tested']>=6).sum()):,} 全R: {int(g.loc[g['tested']>=6,'allR'].sum()):,} 全NS: {int(g.loc[g['tested']>=6,'allNS'].sum()):,}")

# C3 Stanford tobramycin 无效行
sc = pd.read_csv(os.path.join(S, "microbiology_cultures_cohort.csv"), usecols=["organism","antibiotic","susceptibility","was_positive"], low_memory=False, encoding="utf-8-sig")
pa = sc["organism"].astype(str).str.contains("PSEUDOMONAS AERUGINOSA", na=False)
pos = sc["was_positive"].astype(str).str.strip().isin(["1","1.0"])
tob = norm(sc["antibiotic"]).eq("tobramycin")
sus = sc["susceptibility"].astype(str).str.upper().str.strip()
inv = pa & pos & tob & ~sus.isin(["RESISTANT","SUSCEPTIBLE","INTERMEDIATE"])
p(f"[C3 Stanford PA-TOB 无效行] {int(inv.sum()):,}")
p(sc.loc[inv, "susceptibility"].value_counts(dropna=False).head(5).to_string())

# C4 MGB prior_micro NaN 占位
n_tot = 0; n_nan = 0
for chunk in pd.read_csv(os.path.join(M, "prior_micro_deid_tj.csv"), usecols=["organism","antibiotic","CLSI_2022_pheno","anon_id","order_proc_id_coded"], chunksize=2000000, low_memory=False, encoding="utf-8-sig"):
    n_tot += len(chunk)
    allnan = chunk[["organism","antibiotic","CLSI_2022_pheno"]].isna().all(axis=1)
    n_nan += int(allnan.sum())
p(f"[C4 MGB prior_micro] 占位行(全NaN): {n_nan:,} / {n_tot:,} = {n_nan/n_tot:.4f}")

# ============ 汇总 ============
log("汇总")
if fails:
    p(f"契约检查: {len(fails)} 项违规 -> {fails}")
else:
    p("契约检查: 全部通过 (0 违规)")
f.close()
