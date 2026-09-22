# -*- coding: utf-8 -*-
"""
ARMD Step 11 v3 — 特征抽样核对 (时间序规则版, 匹配 step9 v3)
对抽样患者·抽样培养: 用原始表独立重算特征, 与清洗文件比对
检查: F3暴露(精确days_before) / F4既往菌种 / F5既往耐药(时间序+排除同培养) / F7合并症(时间序)
输出: E:\\ARMD\\audit_out\\report_step11_spotcheck.txt
"""
import os
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"; M = os.path.join(BASE, "ARMD-MGB"); S = os.path.join(BASE, "ARMD-Stanford")
CLEAN = os.path.join(BASE, "data", "clean")
OUT = r"E:\ARMD\audit_out"
os.makedirs(OUT, exist_ok=True)
f = open(os.path.join(OUT, "report_step11_spotcheck.txt"), "w", encoding="utf-8")
fails = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n"); f.flush()
def fail(tag, msg): fails.append(tag); p(f"[FAIL] {tag}: {msg}")

import sys
sys.path.insert(0, os.path.dirname(__file__))
from step9_clean import norm, resolve_drug, DRUG_CLASS, COMORB_GROUPS, comorb_group, PROC_KEYS, as_naive

def chunked_filter(path, cols, pred, chunksize=2000000):
    out = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=chunksize, low_memory=False, encoding="utf-8-sig"):
        m = pred(ch)
        if m.any(): out.append(ch[m])
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

# ---------- MGB ----------
p("=" * 80); p("MGB 特征核对 (v3 时间序)"); p("=" * 80)
base = pd.read_csv(os.path.join(CLEAN, "MGB", "site_base_v3.csv"), low_memory=False, encoding="utf-8-sig")
mem = pd.read_csv(os.path.join(CLEAN, "MGB", "task_meropenem.csv"), low_memory=False, encoding="utf-8-sig")
base["time"] = pd.to_datetime(base["time"], errors="coerce")

abx_cols = [c for c in base.columns if c.startswith("abx_carbapenem_")]
base["_abx_total"] = base[abx_cols].sum(axis=1)
pat_abx = base.sort_values("_abx_total", ascending=False)["anon_id"].iloc[0]
pat_com = base.sort_values("comorb_count", ascending=False)["anon_id"].iloc[0]
pats = [pat_abx, pat_com]
for pid in pats:
    p(f"\n--- 患者 {pid} ---")
    rows = base[base["anon_id"] == pid]
    idx = rows[["anon_id","order_proc_id_coded","time"]].rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"})
    # 索引培养时间序集合
    # F3: 重算碳青霉烯 0-30 暴露 (按培养)
    abx = chunked_filter(os.path.join(M, "prior_abx_deid_tj.csv"),
                         ["anon_id","medication_name","last_dose_to_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                         lambda ch: ch["anon_id"].eq(pid))
    abx["att"] = pd.to_datetime(abx["order_time_jittered_utc_shifted"], errors="coerce")
    abx["d"] = pd.to_numeric(abx["last_dose_to_culture"], errors="coerce")
    abx["cls"] = abx["medication_name"].map(resolve_drug).map(DRUG_CLASS)
    abx = abx[abx["cls"].notna() & abx["att"].notna() & abx["d"].notna()]
    abx["ev"] = abx["att"] - pd.to_timedelta(abx["d"], unit="D")
    m3 = abx.merge(idx, on="anon_id")
    m3 = m3[(m3["idx_t"] > m3["ev"]) & (m3["order_proc_id_coded"] != m3["idx_cx"])]
    m3["db"] = (m3["idx_t"] - m3["ev"]).dt.days
    carb = m3[(m3["cls"] == "carbapenem") & (m3["db"] <= 30)].groupby("idx_cx").size()
    # 取该患者一个培养比对
    cx = rows["order_proc_id_coded"].iloc[0]
    exp = int(carb.get(cx, 0)); got = int(rows.iloc[0]["abx_carbapenem_0_30"])
    if exp != got: fail(f"MGB {pid} F3", f"培养{cx} 期望{exp} 实际{got}")
    p(f"  F3 碳青霉烯0-30(培养{cx}): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")

    # F4 既往 Pseudomonas (按培养)
    po = chunked_filter(os.path.join(M, "prior_org_deid_tj.csv"),
                        ["anon_id","prior_org","prior_org_days_to_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                        lambda ch: ch["anon_id"].eq(pid))
    po["att"] = pd.to_datetime(po["order_time_jittered_utc_shifted"], errors="coerce")
    po["d"] = pd.to_numeric(po["prior_org_days_to_culture"], errors="coerce")
    po_p = po[po["prior_org"].astype(str).str.upper().str.contains("PSEUDOMONAS") & po["att"].notna() & po["d"].notna()]
    po_p["ev"] = po_p["att"] - pd.to_timedelta(po_p["d"], unit="D")
    m4 = po_p.merge(idx, on="anon_id")
    m4 = m4[(m4["idx_t"] > m4["ev"]) & (m4["order_proc_id_coded"] != m4["idx_cx"])]
    exp = int(cx in set(m4["idx_cx"])); got = int(rows.iloc[0]["prior_pseudo"])
    if exp != got: fail(f"MGB {pid} F4", f"期望{exp} 实际{got}")
    p(f"  F4 既往Pseudomonas(培养{cx}): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")

    # F7 合并症 (时间序, 按培养)
    cm = chunked_filter(os.path.join(M, "comorbidity_deid_tj.csv"),
                        ["anon_id","category","order_proc_id_coded","order_time_jittered_utc_shifted"],
                        lambda ch: ch["anon_id"].eq(pid))
    cm["att"] = pd.to_datetime(cm["order_time_jittered_utc_shifted"], errors="coerce")
    cm = cm[cm["att"].notna()]
    m7 = cm.merge(idx, on="anon_id")
    m7 = m7[(m7["idx_t"] > m7["att"]) & (m7["order_proc_id_coded"] != m7["idx_cx"])]
    cnt = m7.groupby("idx_cx").size()
    exp = int(cnt.get(cx, 0)); got = int(rows.iloc[0]["comorb_count"])
    if exp != got: fail(f"MGB {pid} F7", f"培养{cx} 期望{exp} 实际{got}")
    p(f"  F7 合并症历史行(培养{cx}): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")
    exp_chf = int(((m7["idx_cx"]==cx) & (m7["category"].map(comorb_group)=="chf")).any())
    got_chf = int(rows.iloc[0]["comorb_chf"])
    if exp_chf != got_chf: fail(f"MGB {pid} F7-chf", f"期望{exp_chf} 实际{got_chf}")
    p(f"  F7 CHF(培养{cx}): 重算={exp_chf} 清洗={got_chf} -> {'OK' if exp_chf==got_chf else 'MISMATCH'}")

    # F5 既往 PA-MEM 耐药 (时间序, 按培养)
    pm = chunked_filter(os.path.join(M, "prior_micro_deid_tj.csv"),
                        ["anon_id","organism","antibiotic","CLSI_2022_pheno","prior_AST_time_to_culture","order_proc_id_coded","order_time_jittered_utc_shifted"],
                        lambda ch: ch["anon_id"].eq(pid) & ch["organism"].astype(str).str.upper().str.startswith("PSEUDOMONAS AERUGINOSA"))
    pm["att"] = pd.to_datetime(pm["order_time_jittered_utc_shifted"], errors="coerce")
    pm["d"] = pd.to_numeric(pm["prior_AST_time_to_culture"], errors="coerce")
    pmR = pm[pm["CLSI_2022_pheno"].astype(str).str.upper().str.strip().isin(["RESISTANT","INTERMEDIATE"])
             & norm(pm["antibiotic"]).eq("meropenem") & pm["att"].notna() & pm["d"].notna()]
    pmR["ev"] = pmR["att"] - pd.to_timedelta(pmR["d"], unit="D")
    m5 = pmR.merge(idx, on="anon_id")
    m5 = m5[(m5["idx_t"] > m5["ev"]) & (m5["order_proc_id_coded"] != m5["idx_cx"])]
    exp = int(cx in set(m5["idx_cx"]))
    memrow = mem[mem["order_proc_id_coded"] == cx]
    got = int(memrow["prior_pa_res"].iloc[0]) if len(memrow) else -99
    if exp != got: fail(f"MGB {pid} F5", f"培养{cx} 期望{exp} 实际{got}")
    p(f"  F5 既往PA-MEM耐药(培养{cx}): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")

# ---------- Stanford (F5 排除 + F7 活动规则) ----------
p("\n" + "=" * 80); p("Stanford 特征核对 (v3 时间序 + 活动规则)"); p("=" * 80)
s_fp = os.path.join(CLEAN, "Stanford", "site_base_v3.csv")
if os.path.exists(s_fp):
    s_base = pd.read_csv(s_fp, low_memory=False, encoding="utf-8-sig")
    s_base["time"] = pd.to_datetime(s_base["time"], errors="coerce")
    pid = s_base["anon_id"].iloc[0]
    p(f"抽样患者: {pid}")
    rows = s_base[s_base["anon_id"] == pid]
    idx = rows[["anon_id","order_proc_id_coded","time"]].rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"})
    cx = rows["order_proc_id_coded"].iloc[0]

    pm = chunked_filter(os.path.join(S, "microbiology_cultures_microbial_resistance.csv"),
                        ["anon_id","organism","antibiotic","resistant_time_to_culturetime","order_proc_id_coded","order_time_jittered_utc"],
                        lambda ch: ch["anon_id"].eq(pid) & ch["organism"].astype(str).str.upper().str.startswith("PSEUDOMONAS AERUGINOSA"))
    pm["att"] = as_naive(pm["order_time_jittered_utc"])
    pm["d"] = pd.to_numeric(pm["resistant_time_to_culturetime"], errors="coerce")
    pmR = pm[norm(pm["antibiotic"]).eq("meropenem") & pm["att"].notna() & pm["d"].notna()]
    pmR["ev"] = pmR["att"] - pd.to_timedelta(pmR["d"], unit="D")
    m5 = pmR.merge(idx, on="anon_id")
    m5 = m5[(m5["idx_t"] > m5["ev"]) & (m5["order_proc_id_coded"] != m5["idx_cx"])]
    exp = int(cx in set(m5["idx_cx"]))
    s_mem = pd.read_csv(os.path.join(CLEAN, "Stanford", "task_meropenem.csv"), low_memory=False, encoding="utf-8-sig")
    got = int(s_mem[s_mem["order_proc_id_coded"] == cx]["prior_pa_res"].iloc[0]) if len(s_mem[s_mem["order_proc_id_coded"] == cx]) else -99
    if exp != got: fail(f"Stanford {pid} F5", f"期望{exp} 实际{got}")
    p(f"  F5 既往PA-MEM耐药(排除同培养+时间序): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")

    cm = chunked_filter(os.path.join(S, "microbiology_cultures_comorbidity.csv"),
                        ["anon_id","comorbidity_component","comorbidity_component_start_days_culture","comorbidity_component_end_days_culture","order_proc_id_coded","order_time_jittered_utc"],
                        lambda ch: ch["anon_id"].eq(pid))
    cm["att"] = as_naive(cm["order_time_jittered_utc"])
    st = pd.to_numeric(cm["comorbidity_component_start_days_culture"], errors="coerce")
    en = pd.to_numeric(cm["comorbidity_component_end_days_culture"], errors="coerce")
    cm = cm[cm["att"].notna() & st.le(0) & (en.isna() | en.lt(0))]
    m7 = cm.merge(idx, on="anon_id")
    m7 = m7[(m7["idx_t"] > m7["att"]) & (m7["order_proc_id_coded"] != m7["idx_cx"])]
    cnt = m7.groupby("idx_cx").size()
    exp = int(cnt.get(cx, 0)); got = int(rows.iloc[0]["comorb_count"])
    if exp != got: fail(f"Stanford {pid} F7", f"期望{exp} 实际{got}")
    p(f"  F7 活动合并症(历史行+活动规则): 重算={exp} 清洗={got} -> {'OK' if exp==got else 'MISMATCH'}")
else:
    p("[SKIP] Stanford 基座未构建(全矩阵重建后补跑)")

p("\n" + "=" * 80)
if fails: p(f"核对结果: {len(fails)} 项不一致 -> {fails}")
else: p("核对结果: 全部一致 (时间序规则下特征与原始表逐项吻合)")
f.close()
print("完成", flush=True)
