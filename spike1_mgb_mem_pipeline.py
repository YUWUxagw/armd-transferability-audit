# -*- coding: utf-8 -*-
"""
ARMD 尖峰测试 #1 — MGB 单站 PA-MEM 任务端到端 pipeline
验证: 队列构建 / 标签规则 / 特征子集(F1-F5+mucoid) / 未来信息扫描 / 患者级折号
输出: E:\\ARMD\\audit_out\\spike1_report.txt + spike1_cohort_sample.csv
"""
import os, time, zlib
import pandas as pd

M = r"E:\ARMD\ARMD-MGB"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
REPORT = os.path.join(OUT, "spike1_report.txt")
f = open(REPORT, "w", encoding="utf-8")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); f.write(line + "\n"); f.flush()

def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n"); f.flush()

def norm(s):
    return (s.astype(str).str.lower()
             .str.replace("/", "_", regex=False)
             .str.replace("-", "_", regex=False)
             .str.strip())

TASK_DRUG = "meropenem"

# ============ 1. 主表扫描 → 队列 ============
log("1. 主表扫描")
cols = ["anon_id","order_proc_id_coded","organism","antibiotic","CLSI_2022_pheno",
        "has_AST","neg_cx","prelim_AST","mult_org_ast","culture_description",
        "order_time_jittered_utc_shifted"]
mgb = pd.read_csv(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"),
                  usecols=cols, low_memory=False, encoding="utf-8-sig")
p(f"主表行数: {len(mgb):,}")

org = mgb["organism"].astype(str).str.upper()
pa = org.str.startswith("PSEUDOMONAS AERUGINOSA")
anorm = norm(mgb["antibiotic"])
task = pa & (anorm == TASK_DRUG)
p(f"PA 行: {int(pa.sum()):,} | PA-MEM 行: {int(task.sum()):,}")

# 排除 preliminary 与阴性
prelim = mgb["prelim_AST"].astype(str).str.strip().eq("X")
neg = mgb["neg_cx"].astype(str).str.strip().eq("X")
sub = mgb[task & ~prelim & ~neg].copy()
p(f"剔除 prelim({int(prelim[task].sum()):,})与阴性({int(neg[task].sum()):,})后: {len(sub):,}")

# 标签
ph = sub["CLSI_2022_pheno"].astype(str).str.upper().str.strip()
R = ph.eq("RESISTANT"); S = ph.eq("SUSCEPTIBLE"); I_ = ph.eq("INTERMEDIATE")
lab = pd.Series(0, index=sub.index)
lab[R] = 1; lab[I_] = 1          # 主规则: I 并入 R
valid = R | S | I_
sub = sub[valid].copy()
sub["label"] = lab[valid].astype(int)
p(f"有效标签行: {len(sub):,} (R={int(sub['label'].sum()):,}) | 无效标签丢弃: {int((~valid).sum()):,}")

# 培养级合并(多重 PA 分离株: 任一 R → R)
sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")
p(f"培养级队列(去重后): {len(sub):,} | 患者数: {sub['anon_id'].nunique():,}")
p(f"培养级耐药率: {sub['label'].mean():.3f} | mult_org_ast 行: {int((sub['mult_org_ast'].astype(str).str.strip().eq('X')).sum()):,}")

# ============ 2. 特征子集 (F1/F2/F3/F4/F5 + mucoid) ============
log("2. 特征构建")
# F9 mucoid
sub["mucoid"] = org[sub.index].str.contains("MUCOID").astype(int)

# F1 人口学
demo = pd.read_csv(os.path.join(M, "demographics_deid_tj.csv"),
                   usecols=["anon_id","age","gender"], low_memory=False, encoding="utf-8-sig")
demo = demo.drop_duplicates("anon_id")
demo["age_bin"] = demo["age"].astype(str).str.replace(" years", "", regex=False)
demo["gender_bin"] = demo["gender"].astype(str).map({"Female": 0, "Male": 1}).fillna(-1).astype(int)
sub = sub.merge(demo[["anon_id","age_bin","gender_bin"]], on="anon_id", how="left")

# F2 ward + specimen
ward = pd.read_csv(os.path.join(M, "ward_type_deid_tj.csv"), low_memory=False, encoding="utf-8-sig")
ward = ward.drop_duplicates("order_proc_id_coded")
def ward_of(r):
    if str(r.get("hosp_ward_IP")) == "1": return "IP"
    if str(r.get("hosp_ward_ER")) == "1": return "ER"
    if str(r.get("hosp_ward_OP")) == "1": return "OP"
    return "other"
ward["ward_h"] = ward.apply(ward_of, axis=1)
sub = sub.merge(ward[["order_proc_id_coded","ward_h"]], on="order_proc_id_coded", how="left")
sub["specimen_h"] = sub["culture_description"].astype(str).str.replace("RESPIRATORY_TRACT", "RESPIRATORY", regex=False)

# F3 既往抗生素暴露 (窗口, drug_class 子集)
abx = pd.read_csv(os.path.join(M, "prior_abx_deid_tj.csv"),
                  usecols=["anon_id","drug_class","last_dose_to_culture"],
                  low_memory=False, encoding="utf-8-sig")
abx["d"] = pd.to_numeric(abx["last_dose_to_culture"], errors="coerce")
abx = abx[abx["d"] > 0]                       # 泄漏扫描: 负值=未来, 剔除并计数
future_abx = int((pd.to_numeric(abx["last_dose_to_culture"], errors="coerce") < 0).sum())
cls = abx["drug_class"].astype(str)
F3_KEYS = {"carbapenem": ["carbapenem"], "fluoroquinolone": ["fluoroquinolone"],
           "aminoglycoside": ["aminoglycoside"], "antipseudomonal_bl": ["extended_spectrum_penicillin"]}
def win(d): return "0-30" if d <= 30 else ("31-90" if d <= 90 else ("91-180" if d <= 180 else ("181-365" if d <= 365 else ">365")))
abx["win"] = abx["d"].map(win)
for k, pats in F3_KEYS.items():
    mk = cls.isin(pats)
    piv = abx[mk].groupby(["anon_id","win"]).size().unstack(fill_value=0)
    for w in ["0-30","31-90","91-180","181-365",">365"]:
        col = f"abx_{k}_{w}"
        sub[col] = sub["anon_id"].map(piv[w] if w in piv.columns else pd.Series(0, index=piv.index)).fillna(0).astype(int)

# F4 既往 Pseudomonas 属检出
po = pd.read_csv(os.path.join(M, "prior_org_deid_tj.csv"),
                 usecols=["anon_id","prior_org","prior_org_days_to_culture"],
                 low_memory=False, encoding="utf-8-sig")
po["d"] = pd.to_numeric(po["prior_org_days_to_culture"], errors="coerce")
po["pseudo"] = po["prior_org"].astype(str).str.upper().str.contains("PSEUDOMONAS")
po_p = po[po["pseudo"] & (po["d"] > 0)]
g = po_p.groupby("anon_id")["d"].min()
sub["prior_pseudo"] = sub["anon_id"].isin(po_p["anon_id"]).astype(int)
sub["prior_pseudo_days"] = sub["anon_id"].map(g)

# F5 既往 PA-MEM 耐药史
pm = pd.read_csv(os.path.join(M, "prior_micro_deid_tj.csv"),
                 usecols=["anon_id","organism","antibiotic","CLSI_2022_pheno","prior_AST_time_to_culture"],
                 low_memory=False, encoding="utf-8-sig")
pm["d"] = pd.to_numeric(pm["prior_AST_time_to_culture"], errors="coerce")
pm_org = pm["organism"].astype(str).str.upper().str.startswith("PSEUDOMONAS AERUGINOSA")
pm_drug = norm(pm["antibiotic"]).eq(TASK_DRUG)
pm_R = pm["CLSI_2022_pheno"].astype(str).str.upper().str.strip().isin(["RESISTANT","INTERMEDIATE"])
pm_hist = pm[pm_org & pm_drug & pm_R & (pm["d"] > 0)]
sub["prior_pa_mem_res"] = sub["anon_id"].isin(pm_hist["anon_id"]).astype(int)

# ============ 3. 未来信息扫描 ============
log("3. 泄漏审计(未来信息扫描)")
fut_org = int((po["d"] < 0).sum())
fut_mic = int((pm["d"] < 0).sum())
p(f"prior_abx 未来值(负): {future_abx:,} | prior_org 未来值: {fut_org:,} | prior_micro 未来值: {fut_mic:,}")
p(f"特征相对字段全部取 >0 过滤; 若未来值占比>0.1% 需在正式清洗中登记处理")

# ============ 4. 患者级 5 折折号 (固定种子, hash 法) ============
log("4. 患者级折号")
patients = sub["anon_id"].unique()
def fold_of(a): return zlib.crc32(str(a).encode()) % 5
fd = {a: fold_of(a) for a in patients}
sub["fold_id"] = sub["anon_id"].map(fd)
p(f"折号分布: {dict(sub['fold_id'].value_counts().sort_index())}")

# ============ 5. 输出 ============
log("5. 输出")
feat_cols = ["order_proc_id_coded","anon_id","label","mucoid","age_bin","gender_bin",
             "ward_h","specimen_h","prior_pseudo","prior_pseudo_days","prior_pa_mem_res","fold_id"]
feat_cols += [c for c in sub.columns if c.startswith("abx_")]
miss = sub[feat_cols].isna().mean()
p("\n特征缺失率:")
p(miss[miss > 0].to_string() if (miss > 0).any() else "全部特征无缺失")
sample = sub[feat_cols].head(1000)
sample.to_csv(os.path.join(OUT, "spike1_cohort_sample.csv"), index=False)
p(f"\n尖峰队列: {len(sub):,} 培养 / {sub['anon_id'].nunique():,} 患者 | 耐药率 {sub['label'].mean():.3f}")
p(f"样本 1000 行已存: {os.path.join(OUT, 'spike1_cohort_sample.csv')}")
log("完成")
f.close()
