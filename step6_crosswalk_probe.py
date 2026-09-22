# -*- coding: utf-8 -*-
"""
ARMD Step 6 — 三站点 crosswalk 采样探查
取各表小样本的真实取值, 为字段谐化登记表提供映射依据
输出: 控制台 + E:\\ARMD\\audit_out\\report_step6_crosswalk.txt
"""
import os
import pandas as pd

M = r"E:\ARMD\ARMD-MGB"
S = r"E:\ARMD\ARMD-Stanford"
U = r"E:\ARMD\ARMD-UTSW"
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
f = open(os.path.join(OUT, "report_step6_crosswalk.txt"), "w", encoding="utf-8")

def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n")

def peek(path, usecols=None, n=2000, tag=""):
    try:
        df = pd.read_csv(path, nrows=n, usecols=usecols, low_memory=False, encoding="utf-8-sig")
        return df
    except Exception as e:
        p(f"[{tag}] 读取失败: {e}")
        return None

# --- 人口学: 年龄/性别形态 ---
p("== 年龄/性别 ==")
for tag, path in [("MGB", os.path.join(M, "demographics_deid_tj.csv")),
                  ("Stanford", os.path.join(S, "microbiology_cultures_demographics.csv")),
                  ("UTSW", os.path.join(U, "microbiology_cultures_demographics.csv"))]:
    df = peek(path, ["age", "gender"], tag=tag)
    if df is not None:
        p(f"[{tag}] age 样例: {df['age'].astype(str).head(5).tolist()} | gender 取值: {dict(df['gender'].astype(str).value_counts().head(3))}")

# --- MGB 主表: 标本来源/就诊模式粒度 ---
p("\n== 标本来源与就诊模式 ==")
df = peek(os.path.join(M, "microbiology_cohort_deid_tj_updated.csv"), ["culture_description", "ordering_mode"])
if df is not None:
    p("[MGB] culture_description top12:")
    p(df["culture_description"].value_counts().head(12).to_string())
    p("[MGB] ordering_mode:")
    p(df["ordering_mode"].value_counts(dropna=False).head(8).to_string())
df = peek(os.path.join(S, "microbiology_cultures_cohort.csv"), ["ordering_mode"])
if df is not None:
    p("\n[Stanford] ordering_mode:")
    p(df["ordering_mode"].value_counts(dropna=False).head(8).to_string())
df = peek(os.path.join(U, "microbiology_cultures_cohort.csv"), ["ordering_mode"])
if df is not None:
    p("\n[UTSW] ordering_mode:")
    p(df["ordering_mode"].value_counts(dropna=False).head(8).to_string())

# --- 既往抗生素暴露: 类别词表 ---
p("\n== 抗生素类别词表 ==")
df = peek(os.path.join(M, "prior_abx_deid_tj.csv"), ["drug_class"], n=50000)
if df is not None:
    p("[MGB prior_abx] drug_class top20:")
    p(df["drug_class"].value_counts(dropna=False).head(20).to_string())
for tag, path in [("Stanford", os.path.join(S, "microbiology_cultures_antibiotic_class_exposure.csv")),
                  ("UTSW", os.path.join(U, "microbiology_cultures_antibiotic_class_exposure.csv"))]:
    df = peek(path, ["antibiotic_class"], n=50000)
    if df is not None:
        p(f"[{tag} antibiotic_class_exposure] antibiotic_class top20:")
        p(df["antibiotic_class"].value_counts(dropna=False).head(20).to_string())

# --- 既往菌种: 命名对照 ---
p("\n== 既往菌种命名 ==")
df = peek(os.path.join(M, "prior_org_deid_tj.csv"), ["prior_org"], n=50000)
if df is not None:
    p("[MGB prior_org] top12:")
    p(df["prior_org"].value_counts().head(12).to_string())
for tag, path in [("Stanford", os.path.join(S, "microbiology_culture_prior_infecting_organism.csv")),
                  ("UTSW", os.path.join(U, "microbiology_cultures_prior_infecting_organism.csv"))]:
    df = peek(path, ["prior_organism"], n=50000)
    if df is not None:
        p(f"[{tag} prior_infecting_organism] top12:")
        p(df["prior_organism"].value_counts().head(12).to_string())

# --- 合并症词表 ---
p("\n== 合并症词表 ==")
df = peek(os.path.join(M, "comorbidity_deid_tj.csv"), ["category"], n=50000)
if df is not None:
    p("[MGB comorbidity] category top12:")
    p(df["category"].value_counts(dropna=False).head(12).to_string())
df = peek(os.path.join(S, "microbiology_cultures_comorbidity.csv"), ["comorbidity_component"], n=50000)
if df is not None:
    p("[Stanford comorbidity] comorbidity_component top12:")
    p(df["comorbidity_component"].value_counts(dropna=False).head(12).to_string())
df = peek(os.path.join(U, "microbiology_cultures_comorbidity.csv"), ["comorbidity_component"], n=50000)
if df is not None:
    p("[UTSW comorbidity] comorbidity_component top12:")
    p(df["comorbidity_component"].value_counts(dropna=False).head(12).to_string())

# --- ward 标志取值 ---
p("\n== ward 标志 ==")
df = peek(os.path.join(M, "ward_type_deid_tj.csv"))
if df is not None:
    wc = [c for c in df.columns if c.startswith("hosp_ward")]
    p(f"[MGB ward_type] 标志列: {wc}")
    for c in wc:
        p(f"  {c}: {dict(df[c].astype(str).value_counts(dropna=False).head(3))}")
df = peek(os.path.join(S, "microbiology_cultures_ward_info.csv"))
if df is not None:
    wc = [c for c in df.columns if c.startswith("hosp_ward")]
    p(f"[Stanford ward_info] 标志列: {wc}")
    for c in wc:
        p(f"  {c}: {dict(df[c].astype(str).value_counts(dropna=False).head(3))}")

# --- 既往微生物史(药敏): 形态 ---
p("\n== 既往微生物药敏史 ==")
df = peek(os.path.join(M, "prior_micro_deid_tj.csv"), ["organism", "antibiotic", "CLSI_2022_pheno"], n=20000)
if df is not None:
    p("[MGB prior_micro] 样例:")
    p(df.head(3).to_string())
    p(f"[MGB prior_micro] CLSI 取值: {dict(df['CLSI_2022_pheno'].astype(str).value_counts().head(6))}")
df = peek(os.path.join(S, "microbiology_cultures_microbial_resistance.csv"), ["organism", "antibiotic"], n=20000)
if df is not None:
    p("\n[Stanford microbial_resistance] 样例:")
    p(df.head(3).to_string())

p("\n完成")
f.close()
