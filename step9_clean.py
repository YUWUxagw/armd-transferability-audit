# -*- coding: utf-8 -*-
"""
ARMD Step 9 v3.1 — 三站参数化清洗 pipeline (培养级精确 point-in-time, 培养对架构)
v3 修正 (2026-08-07 特征核对驱动): 全部历史特征按时间序构建
v3.1: 患者内培养对(att<idx,gap)替代多对多 merge, 消除内存爆炸
用法: python step9_clean.py [site] [task]
"""
import os, sys, time, zlib, re, hashlib
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
SITES = {
    "MGB": dict(dir=os.path.join(BASE, "ARMD-MGB"),
        cohort="microbiology_cohort_deid_tj_updated.csv",
        label_col="CLSI_2022_pheno", pos_col=None, neg_col="neg_cx", prelim_col="prelim_AST",
        ward_icu=None, time_col="order_time_jittered_utc_shifted",
        demo=("demographics_deid_tj.csv","age","gender","Female","Male"),
        ward="ward_type_deid_tj.csv",
        abx=("prior_abx_deid_tj.csv","medication_name","last_dose_to_culture","order_proc_id_coded"),
        priorg=("prior_org_deid_tj.csv","prior_org","prior_org_days_to_culture","order_proc_id_coded"),
        prmic=("prior_micro_deid_tj.csv","organism","antibiotic","CLSI_2022_pheno","prior_AST_time_to_culture","order_proc_id_coded"),
        comorb=("comorbidity_deid_tj.csv","category",None,None,"order_proc_id_coded"),
        proc=("prior_procedures_deid_tj.csv","procedure_description","procedure_days_culture","order_proc_id_coded"),
        nh=("nursing_home_visits_deid_tj.csv","nursing_home_visit_culture","order_proc_id_coded"),
        adi=("ADI_deid_tj.csv","adi_score","order_proc_id_coded")),
    "Stanford": dict(dir=os.path.join(BASE, "ARMD-Stanford"),
        cohort="microbiology_cultures_cohort.csv",
        label_col="susceptibility", pos_col="was_positive", neg_col=None, prelim_col=None,
        ward_icu="hosp_ward_ICU", time_col="order_time_jittered_utc",
        demo=("microbiology_cultures_demographics.csv","age","gender",None,None),
        ward="microbiology_cultures_ward_info.csv",
        abx=("microbiology_cultures_prior_med.csv","medication_name","medication_time_to_culturetime","order_proc_id_coded"),
        priorg=("microbiology_culture_prior_infecting_organism.csv","prior_organism","prior_infecting_organism_days_to_culutre","order_proc_id_coded"),
        prmic=("microbiology_cultures_microbial_resistance.csv","organism","antibiotic",None,"resistant_time_to_culturetime","order_proc_id_coded"),
        comorb=("microbiology_cultures_comorbidity.csv","comorbidity_component","comorbidity_component_start_days_culture","comorbidity_component_end_days_culture","order_proc_id_coded"),
        proc=("microbiology_cultures_priorprocedures.csv","procedure_description","procedure_time_to_culturetime","order_proc_id_coded"),
        nh=("microbiology_cultures_nursing_home_visits.csv","nursing_home_visit_culture","order_proc_id_coded"),
        adi=("microbiology_cultures_adi_scores.csv","adi_score","order_proc_id_coded")),
    "UTSW": dict(dir=os.path.join(BASE, "ARMD-UTSW"),
        cohort="microbiology_cultures_cohort.csv",
        label_col="susceptibility", pos_col="was_positive", neg_col=None, prelim_col=None,
        ward_icu="hosp_ward_ICU", time_col="order_time_jittered",
        demo=("microbiology_cultures_demographics.csv","age","gender",None,None),
        ward="microbiology_cultures_ward_info.csv",
        abx=("microbiology_cultures_prior_med.csv","medication_name","medication_time_to_culturetime","order_proc_id_coded"),
        priorg=("microbiology_cultures_prior_infecting_organism.csv","prior_organism","prior_infecting_organism_days_to_culture","order_proc_id_coded"),
        prmic=("microbiology_cultures_microbial_resistance.csv","organism","antibiotic",None,"resistant_time_to_culturetime","order_proc_id_coded"),
        comorb=("microbiology_cultures_comorbidity.csv","comorbidity_component","comorbidity_component_start_days_culture","comorbidity_component_end_days_culture","order_proc_id_coded"),
        proc=("microbiology_cultures_prior_procedures.csv","procedure_description","procedure_time_to_culturetime","order_proc_id_coded"),
        nh=("microbiology_cultures_nursing_home_visits.csv","nursing_home_visit_culture","order_proc_id_coded"),
        adi=("microbiology_cultures_adi_scores.csv","adi_score","order_proc_id_coded")),
}
TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime",
         "piperacillin_tazobactam","aztreonam","amikacin","tobramycin"]
WIN = [("0_30",30),("31_90",90),("91_180",180),("181_365",365),("gt365",1e9)]
CLASSES = ["carbapenem","fluoroquinolone","aminoglycoside","antipseudomonal_bl","glycopeptide"]
OUTDIR = os.path.join(BASE, "data", "clean")
CHUNK = 2000000

def load_dict(name):
    df = pd.read_csv(os.path.join(BASE, "project", "dicts", name))
    return dict(zip(df["drug"], df["canonical_class"])) if "drug" in df.columns else df
DRUG_CLASS = load_dict("dict_canonical_drugs.csv")

STRIP = {"dextrs","dextrose","hcl","hci","nacl","hyclate","monohydrate","macrocrystal","axetil",
         "tromethamine","sodium_citrate","pot","in","sodium_chloride","d5w","ns","bacteriostatic",
         "pack","phosphate","ethylsuccinate","dexamethasone","hippurate","mandelate","citrate",
         "sulfate","sodium","diluent","combo","chloride","iso","sulfa","tromethamin"}
ALIAS = {"zithromax":"azithromycin","bactrim_ds":"trimethoprim_sulfamethoxazole","bactrim":"trimethoprim_sulfamethoxazole",
         "cipro":"ciprofloxacin","augmentin":"amoxicillin_clavulanate","keflex":"cephalexin",
         "macrobid":"nitrofurantoin","macrodantin":"nitrofurantoin","levaquin":"levofloxacin",
         "xifaxan":"rifaximin","flagyl":"metronidazole","hiprex":"methenamine","zyvox":"linezolid",
         "sulfamethoxazole_trimethoprim":"trimethoprim_sulfamethoxazole"}
def norm(s):
    return (s.astype(str).str.lower().str.replace("/","_",regex=False)
             .str.replace("-","_",regex=False).str.replace(r"[()]","_",regex=False).str.strip())
def resolve_drug(name):
    d = norm(pd.Series([name]))[0]
    if d in DRUG_CLASS: return d
    if d in ALIAS: return ALIAS[d]
    toks = [t for t in re.split(r"[_\s]+", d) if t and t not in STRIP]
    d2 = "_".join(toks)
    if d2 in DRUG_CLASS: return d2
    if d2 in ALIAS: return ALIAS[d2]
    return None

COMORB_GROUPS = [("chf",["heart failure"]),("ckd",["kidney","renal"]),("diabetes",["diabetes"]),
    ("copd",["copd","chronic obstructive"]),("malignancy",["cancer","malign","neoplasm","leukemia","lymphoma","melanoma"]),
    ("metastatic",["metastatic"]),("liver",["liver","hepatic","cirrhosis"]),
    ("immunocompromised",["immun","transplant","hiv","aids"]),("obesity",["obesity","overweight"]),
    ("dementia",["dementia","alzheimer"]),("stroke",["stroke","cerebrovascular","hemiplegia"]),
    ("coagulopathy",["coagul"]),("fluid_electrolyte",["fluid and electrolyte"]),("anemia",["anemia"])]
def comorb_group(c):
    c = str(c).lower()
    for g, kws in COMORB_GROUPS:
        if any(k in c for k in kws): return g
    return "other"
PROC_KEYS = {"mechvent":["mechvent","mechanical ventil"],"cvc":["cvc","central venous","central line"],
             "urinary_cath":["urethral_catheter","urinary catheter","foley","urethral"],"dialysis":["dialysis","dialys"],
             "surgery":["surgical_procedure","surgery","operative","operation"]}

def read_chunked(path, cols, filt=None, pa_org_col=None):
    out = []
    for ch in pd.read_csv(path, usecols=cols, chunksize=CHUNK, low_memory=False, encoding="utf-8-sig"):
        if pa_org_col:
            ch = ch[ch[pa_org_col].astype(str).str.upper().str.startswith("PSEUDOMONAS AERUGINOSA")]
        if filt:
            ch = ch[filt(ch)]
        if len(ch): out.append(ch)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame(columns=cols)

def as_naive(s):
    s = pd.to_datetime(s, errors="coerce")
    if getattr(s.dt, "tz", None) is not None:
        s = s.dt.tz_localize(None)
    return s

def build_site_base(site_name):
    cf = SITES[site_name]; d = cf["dir"]; t0 = time.time()
    print(f"[{site_name}] 阶段A(v3.1) 开始", flush=True)
    cached = os.path.join(OUTDIR, site_name, "site_base_v3.csv")
    if os.path.exists(cached):
        print(f"[{site_name}] 阶段A 命中缓存", flush=True); return pd.read_csv(cached)

    head = pd.read_csv(os.path.join(d, cf["cohort"]), nrows=0)
    use = [c for c in ["anon_id","order_proc_id_coded","organism","antibiotic",cf["label_col"],
                       "culture_description", cf["time_col"]]
           + ([cf["pos_col"]] if cf["pos_col"] else []) + ([cf["neg_col"]] if cf["neg_col"] else [])
           + ([cf["prelim_col"]] if cf["prelim_col"] else []) if c in head.columns]
    mc = pd.read_csv(os.path.join(d, cf["cohort"]), usecols=use, low_memory=False, encoding="utf-8-sig")
    org = mc["organism"].astype(str).str.upper()
    sub = mc[org.str.startswith("PSEUDOMONAS AERUGINOSA") & norm(mc["antibiotic"]).isin(TASKS)].copy()
    if cf["prelim_col"]: sub = sub[~sub[cf["prelim_col"]].astype(str).str.strip().eq("X")]
    if cf["neg_col"]:    sub = sub[~sub[cf["neg_col"]].astype(str).str.strip().eq("X")]
    if cf["pos_col"]:    sub = sub[sub[cf["pos_col"]].astype(str).str.strip().isin(["1","1.0"])]
    ph = sub[cf["label_col"]].astype(str).str.upper().str.strip()
    R = ph.eq("RESISTANT"); S = ph.eq("SUSCEPTIBLE"); I = ph.eq("INTERMEDIATE")
    lab = pd.Series(0, index=sub.index); lab[R | I] = 1
    sub = sub[R | S | I].copy(); sub["label"] = lab[R | S | I].astype(int)
    sub["drug"] = norm(sub["antibiotic"])
    sub["mucoid"] = org[sub.index].str.contains("MUCOID").astype(int)
    sub["specimen"] = sub["culture_description"].astype(str).str.replace("RESPIRATORY_TRACT","RESPIRATORY",regex=False)
    print(f"[{site_name}] PA任务行 {len(sub):,} | 培养 {sub['order_proc_id_coded'].nunique():,} | 患者 {sub['anon_id'].nunique():,}", flush=True)

    cu = sub.drop_duplicates("order_proc_id_coded")[["anon_id","order_proc_id_coded","mucoid","specimen"]].copy()
    cu["time"] = as_naive(sub.drop_duplicates("order_proc_id_coded")[cf["time_col"]])
    patients = set(sub["anon_id"])

    # 历史事件时间 = 挂接时间 - 相对天数 (无相对天数表 = 挂接时间)
    # 统一规则: 事件时间严格早于索引培养时间 (同培养零天行自动排除)
    def hist_events(hist, rel_col, oid_col, carry=None):
        att = as_naive(hist[cf["time_col"]])
        if rel_col:
            d = pd.to_numeric(hist[rel_col], errors="coerce")
            ev = att - pd.to_timedelta(d, unit="D")
        else:
            ev = att
        h = pd.DataFrame({"anon_id": hist["anon_id"].values, "ev": ev.values})
        if carry:
            h = h.join(hist[carry].reset_index(drop=True))
        h = h[h["ev"].notna()]
        m = h.merge(cu[["anon_id","order_proc_id_coded","time"]]
                    .rename(columns={"order_proc_id_coded":"idx_cx","time":"idx_t"}), on="anon_id")
        m = m[m["idx_t"] > m["ev"]]          # 事件严格早于索引培养
        m["days_before"] = (m["idx_t"] - m["ev"]).dt.days
        return m

    demo_fn, age_c, gen_c, f_v, m_v = cf["demo"]
    demo = pd.read_csv(os.path.join(d, demo_fn), usecols=["anon_id",age_c,gen_c], low_memory=False, encoding="utf-8-sig").drop_duplicates("anon_id")
    demo["age_bin"] = demo[age_c].astype(str).str.replace(" years","",regex=False)
    demo["gender_bin"] = (demo[gen_c].astype(str).map({f_v:0,m_v:1}).fillna(-1).astype(int) if f_v
                          else pd.to_numeric(demo[gen_c],errors="coerce").fillna(-1).astype(int))
    cu = cu.merge(demo[["anon_id","age_bin","gender_bin"]], on="anon_id", how="left")

    ward = pd.read_csv(os.path.join(d, cf["ward"]), low_memory=False, encoding="utf-8-sig").drop_duplicates("order_proc_id_coded")
    def ward_of(r):
        if str(r.get("hosp_ward_IP"))=="1": return "IP"
        if str(r.get("hosp_ward_ER"))=="1": return "ER"
        if str(r.get("hosp_ward_OP"))=="1": return "OP"
        return "other"
    ward["ward_h"] = ward.apply(ward_of, axis=1)
    cu = cu.merge(ward[["order_proc_id_coded","ward_h"]], on="order_proc_id_coded", how="left")
    if cf["ward_icu"]:
        icu = ward[["order_proc_id_coded",cf["ward_icu"]]].rename(columns={cf["ward_icu"]:"icu"})
        icu["icu"] = icu["icu"].astype(str).str.strip().eq("1").astype(int)
        cu = cu.merge(icu, on="order_proc_id_coded", how="left"); cu["icu"] = cu["icu"].fillna(0).astype(int)
    else:
        cu["icu"] = 0

    # F3 既往抗生素 (培养级, 时间序; -1 哨兵排除)
    abx_fn, med_c, rel_c, oid_c = cf["abx"]
    abx = read_chunked(os.path.join(d, abx_fn), ["anon_id",med_c,rel_c,oid_c,cf["time_col"]],
                       filt=lambda ch: ch["anon_id"].isin(patients))
    abx["cls"] = abx[med_c].map(resolve_drug).map(DRUG_CLASS)
    abx = abx[abx["cls"].notna() & (pd.to_numeric(abx[rel_c], errors="coerce") > 0)]
    ev3 = hist_events(abx, rel_c, oid_c, carry=["cls"])
    ev3["win"] = ev3["days_before"].apply(lambda x: "0_30" if x<=30 else ("31_90" if x<=90 else ("91_180" if x<=180 else ("181_365" if x<=365 else "gt365"))))
    for cls in CLASSES:
        m = ev3[ev3["cls"]==cls]
        if len(m):
            piv = m.groupby(["idx_cx","win"]).size().unstack(fill_value=0)
            for w,_ in WIN:
                cu[f"abx_{cls}_{w}"] = cu["order_proc_id_coded"].map(piv[w] if w in piv.columns else pd.Series(0,index=piv.index)).fillna(0).astype(int)
        else:
            for w,_ in WIN: cu[f"abx_{cls}_{w}"] = 0

    # F4 既往 Pseudomonas (培养级)
    po_fn, po_c, po_r, po_oid = cf["priorg"]
    po = read_chunked(os.path.join(d, po_fn), ["anon_id",po_c,po_r,po_oid,cf["time_col"]],
                      filt=lambda ch: ch["anon_id"].isin(patients))
    po_p = po[po[po_c].astype(str).str.upper().str.contains("PSEUDOMONAS")]
    ev4 = hist_events(po_p, po_r, po_oid)
    g = ev4.groupby("idx_cx")["days_before"].min()
    cu["prior_pseudo"] = cu["order_proc_id_coded"].isin(set(g.index)).astype(int)
    cu["prior_pseudo_days"] = cu["order_proc_id_coded"].map(g)

    # F5 既往 PA-药耐药 (培养级, 9 药一次)
    pm_fn, pm_org, pm_abx, pm_lab, pm_t, pm_oid = cf["prmic"]
    pm_cols = ["anon_id",pm_org,pm_abx,pm_t,pm_oid,cf["time_col"]] + ([pm_lab] if pm_lab else [])
    pm = read_chunked(os.path.join(d, pm_fn), pm_cols, pa_org_col=pm_org)
    if pm_lab:
        pm = pm[pm[pm_lab].astype(str).str.upper().str.strip().isin(["RESISTANT","INTERMEDIATE"])]
    pm["drug"] = norm(pm[pm_abx])
    pm = pm[pm["drug"].isin(TASKS)]
    ev5 = hist_events(pm, pm_t, pm_oid, carry=["drug"])
    for t in TASKS:
        m = ev5[ev5["drug"]==t]
        cu[f"prior_pa_res_{t}"] = cu["order_proc_id_coded"].isin(set(m["idx_cx"])).astype(int)

    # F6 器械 (培养级)
    pr_fn, pr_c, pr_r, pr_oid = cf["proc"]
    pr = read_chunked(os.path.join(d, pr_fn), ["anon_id",pr_c,pr_r,pr_oid,cf["time_col"]],
                      filt=lambda ch: ch["anon_id"].isin(patients))
    ev6 = hist_events(pr, pr_r, pr_oid, carry=[pr_c])
    for k, kws in PROC_KEYS.items():
        m = ev6[ev6[pr_c].astype(str).str.lower().str.contains("|".join(kws), regex=True)]
        cu[f"proc_{k}"] = cu["order_proc_id_coded"].isin(set(m["idx_cx"])).astype(int)

    # F7 合并症 (培养级, 严格更早 gap>=1; S/U 活动规则)
    cm_fn, cm_c, cm_s, cm_e, cm_oid = cf["comorb"]
    cm_cols = ["anon_id", cm_oid, cf["time_col"]] + ([cm_c] if cm_c else []) + ([cm_s] if cm_s else []) + ([cm_e] if cm_e else [])
    cm = read_chunked(os.path.join(d, cm_fn), cm_cols, filt=lambda ch: ch["anon_id"].isin(patients))
    if cm_s:
        st = pd.to_numeric(cm[cm_s], errors="coerce")
        en = pd.to_numeric(cm[cm_e], errors="coerce") if cm_e else None
        act = st.le(0) & (en.isna() | en.lt(0)) if en is not None else st.le(0)
        cm = cm[act]
    ev7 = hist_events(cm, None, cm_oid, carry=[cm_c])
    ev7["group"] = ev7[cm_c].map(comorb_group)
    for g,_ in COMORB_GROUPS:
        m = ev7[ev7["group"]==g]
        cu[f"comorb_{g}"] = cu["order_proc_id_coded"].isin(set(m["idx_cx"])).astype(int)
    cu["comorb_other"] = cu["order_proc_id_coded"].isin(set(ev7[ev7["group"]=="other"]["idx_cx"])).astype(int)
    cu["comorb_count"] = cu["order_proc_id_coded"].map(ev7.groupby("idx_cx").size()).fillna(0).astype(int)

    # F8 ADI + 养老院
    adi_fn, adi_c, adi_oid = cf["adi"]
    adi = pd.read_csv(os.path.join(d, adi_fn), usecols=["anon_id",adi_c], low_memory=False, encoding="utf-8-sig").drop_duplicates("anon_id")
    adi[adi_c] = pd.to_numeric(adi[adi_c], errors="coerce")
    cu = cu.merge(adi.rename(columns={adi_c:"adi"}), on="anon_id", how="left")
    nh_fn, nh_c, nh_oid = cf["nh"]
    nh = read_chunked(os.path.join(d, nh_fn), ["anon_id",nh_c,nh_oid,cf["time_col"]],
                      filt=lambda ch: ch["anon_id"].isin(patients))
    ev8 = hist_events(nh, nh_c, nh_oid)
    cu["nh_30d"] = cu["order_proc_id_coded"].isin(set(ev8[ev8["days_before"].le(30)]["idx_cx"])).astype(int)

    cu["fold_id"] = cu["anon_id"].map(lambda a: zlib.crc32(str(a).encode()) % 5)
    os.makedirs(os.path.join(OUTDIR, site_name), exist_ok=True)
    cu.to_csv(cached, index=False)
    print(f"[{site_name}] 阶段A(v3.1) 完成: 基座 {len(cu):,} 培养 | 耗时 {time.time()-t0:.0f}s", flush=True)
    return cu

def build_task(site_name, task, base):
    cf = SITES[site_name]; d = cf["dir"]; t0 = time.time()
    cache_fp = os.path.join(OUTDIR, site_name, f"task_{task}.csv")
    if os.path.exists(cache_fp):
        print(f"[{site_name}::{task}] 命中缓存", flush=True); return

    head = pd.read_csv(os.path.join(d, cf["cohort"]), nrows=0)
    use = [c for c in ["anon_id","order_proc_id_coded","organism","antibiotic",cf["label_col"]]
           + ([cf["pos_col"]] if cf["pos_col"] else []) + ([cf["neg_col"]] if cf["neg_col"] else [])
           + ([cf["prelim_col"]] if cf["prelim_col"] else []) if c in head.columns]
    mc = pd.read_csv(os.path.join(d, cf["cohort"]), usecols=use, low_memory=False, encoding="utf-8-sig")
    org = mc["organism"].astype(str).str.upper()
    sub = mc[org.str.startswith("PSEUDOMONAS AERUGINOSA") & norm(mc["antibiotic"]).eq(task)].copy()
    if cf["prelim_col"]: sub = sub[~sub[cf["prelim_col"]].astype(str).str.strip().eq("X")]
    if cf["neg_col"]:    sub = sub[~sub[cf["neg_col"]].astype(str).str.strip().eq("X")]
    if cf["pos_col"]:    sub = sub[sub[cf["pos_col"]].astype(str).str.strip().isin(["1","1.0"])]
    ph = sub[cf["label_col"]].astype(str).str.upper().str.strip()
    R = ph.eq("RESISTANT"); S = ph.eq("SUSCEPTIBLE"); I = ph.eq("INTERMEDIATE")
    lab = pd.Series(0, index=sub.index); lab[R | I] = 1
    sub = sub[R | S | I].copy(); sub["label"] = lab[R | S | I].astype(int)
    sub = sub.sort_values("label", ascending=False).drop_duplicates("order_proc_id_coded")

    out = sub[["order_proc_id_coded","anon_id","label"]].merge(
        base.drop(columns=["anon_id"]), on="order_proc_id_coded", how="left")
    out[f"prior_pa_res_{task}"] = out[f"prior_pa_res_{task}"].fillna(0).astype(int)
    out = out.rename(columns={f"prior_pa_res_{task}": "prior_pa_res"})
    out.to_csv(cache_fp, index=False)
    h = hashlib.sha256(open(cache_fp,"rb").read()).hexdigest()[:12]
    print(f"[{site_name}::{task}] 完成: {len(out):,} 培养 / R率 {out['label'].mean():.3f} | {h} | {time.time()-t0:.0f}s", flush=True)

if __name__ == "__main__":
    site_arg = sys.argv[1] if len(sys.argv) > 1 else None
    task_arg = sys.argv[2] if len(sys.argv) > 2 else None
    sites = [site_arg] if site_arg else list(SITES.keys())
    tasks = [task_arg] if task_arg else TASKS
    for s in sites:
        base = build_site_base(s)
        for t in tasks:
            build_task(s, t, base)
    print("全部完成", flush=True)
