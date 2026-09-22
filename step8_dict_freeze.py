# -*- coding: utf-8 -*-
"""
ARMD Step 8 — 字典冻结草稿生成
1) 规范药物→类别字典 (67 药显式映射 + S/U medication_name 匹配覆盖)
2) 属级映射 (MGB prior_org 35 值)
3) 合并症词汇交集分析 (MGB 508 vs S/U 519/512)
4) 年龄分箱统一表
输出: E:\\ARMD\\project\\dicts\\*.csv + report_step8_dicts.txt
"""
import os, time
import pandas as pd
from collections import Counter

M = r"E:\ARMD\ARMD-MGB"; S = r"E:\ARMD\ARMD-Stanford"; U = r"E:\ARMD\ARMD-UTSW"
VOCAB = r"E:\ARMD\project\vocab"; DICTS = r"E:\ARMD\project\dicts"
os.makedirs(DICTS, exist_ok=True)
REPORT = os.path.join(r"E:\ARMD\audit_out", "report_step8_dicts.txt")
f = open(REPORT, "w", encoding="utf-8")

def log(msg):
    line = f"[{time.strftime('%H:%M:%S')}] {msg}"
    print(line, flush=True); f.write(line + "\n"); f.flush()
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); f.write(line + "\n"); f.flush()
def norm(s):
    return (s.astype(str).str.lower().str.replace("/", "_", regex=False)
             .str.replace("-", "_", regex=False).str.strip())

# ============ 1. 规范药物→类别 ============
log("1. 药物→类别字典")
DRUG_CLASS = {
 # 碳青霉烯
 "meropenem":"carbapenem","ertapenem":"carbapenem","imipenem":"carbapenem",
 "meropenem_vaborbactam":"carbapenem","imipenem_relebactam":"carbapenem",
 # 氟喹诺酮
 "ciprofloxacin":"fluoroquinolone","levofloxacin":"fluoroquinolone",
 "moxifloxacin":"fluoroquinolone","delafloxacin":"fluoroquinolone",
 # 氨基糖苷
 "gentamicin":"aminoglycoside","tobramycin":"aminoglycoside","amikacin":"aminoglycoside",
 # 抗假单胞菌 β-内酰胺
 "cefepime":"antipseudomonal_bl","ceftazidime":"antipseudomonal_bl",
 "piperacillin_tazobactam":"antipseudomonal_bl","aztreonam":"antipseudomonal_bl",
 "ceftazidime_avibactam":"antipseudomonal_bl","ceftolozane_tazobactam":"antipseudomonal_bl",
 "cefiderocol":"antipseudomonal_bl",
 # 其他 β-内酰胺
 "ceftriaxone":"other_beta_lactam","cefazolin":"other_beta_lactam",
 "cephalexin":"other_beta_lactam","cefpodoxime":"other_beta_lactam",
 "cefuroxime":"other_beta_lactam","cefadroxil":"other_beta_lactam",
 "cefdinir":"other_beta_lactam","cefoxitin":"other_beta_lactam",
 "cefotaxime":"other_beta_lactam","cefixime":"other_beta_lactam",
 "cefotetan":"other_beta_lactam","ceftaroline":"other_beta_lactam",
 "ampicillin_sulbactam":"other_beta_lactam","amoxicillin_clavulanate":"other_beta_lactam",
 "ampicillin":"other_beta_lactam","amoxicillin":"other_beta_lactam",
 "penicillin":"other_beta_lactam","oxacillin":"other_beta_lactam",
 "nafcillin":"other_beta_lactam","dicloxacillin":"other_beta_lactam",
 # 糖肽类
 "vancomycin":"glycopeptide","dalbavancin":"glycopeptide",
 # 大环内酯
 "azithromycin":"macrolide","clarithromycin":"macrolide",
 # 四环素
 "doxycycline":"tetracycline","minocycline":"tetracycline","tetracycline":"tetracycline",
 "tigecycline":"tetracycline","omadacycline":"tetracycline","eravacycline":"tetracycline",
 # 磺胺/甲氧苄啶
 "trimethoprim_sulfamethoxazole":"sulfonamide","trimethoprim":"sulfonamide",
 # 抗厌氧
 "metronidazole":"anti_anaerobe",
 # 唑类/棘白菌素/多烯 (抗真菌)
 "fluconazole":"azole","voriconazole":"azole","posaconazole":"azole","itraconazole":"azole",
 "micafungin":"echinocandin","caspofungin":"echinocandin",
 "amphotericin_b":"polyene","amphotericin":"polyene",
 # 其他
 "clindamycin":"lincosamide","colistin":"polymyxin",
 "linezolid":"oxazolidinone","tedizolid":"oxazolidinone",
 "daptomycin":"lipopeptide","fosfomycin":"fosfomycin",
 "nitrofurantoin":"nitrofuran",
}
ANTIPSEUDO = {"cefepime","ceftazidime","piperacillin_tazobactam","aztreonam",
              "ceftazidime_avibactam","ceftolozane_tazobactam","cefiderocol",
              "meropenem","imipenem","ciprofloxacin","levofloxacin",
              "amikacin","tobramycin"}

mgb_names = pd.read_csv(os.path.join(VOCAB, "vocab_MGB_drug_names.csv"))
rows = []
unmatched = []
for _, r in mgb_names.iterrows():
    d = norm(pd.Series([r["value"]]))[0]
    cls = DRUG_CLASS.get(d)
    if cls is None:
        unmatched.append((r["value"], int(r["count"])))
        cls = "other"
    rows.append(dict(drug=d, drug_display=r["value"], canonical_class=cls,
                     antipseudomonal=int(d in ANTIPSEUDO), count=int(r["count"])))
p(f"规范字典覆盖: {len(rows)-len(unmatched)}/{len(rows)} 药 | 未映射: {unmatched if unmatched else '无'}")

# S/U medication_name 匹配（配方后缀剥离 + 别名 + 补充药）
EXTRA_DRUGS = {"rifaximin":"rifamycin","erythromycin":"macrolide",
               "ofloxacin":"fluoroquinolone","rifampin":"rifamycin",
               "rifabutin":"rifamycin","dapsone":"sulfone",
               "ethambutol":"antitubercular","isoniazid":"antitubercular",
               "silver_sulfadiazine":"sulfonamide","sulfadiazine":"sulfonamide",
               "polymyxin_b":"polymyxin","colistimethate":"polymyxin",
               "mupirocin":"other","azithromycin_pack":"macrolide",
               "gatifloxacin":"fluoroquinolone","methenamine":"urinary_antiseptic",
               "methenamine_mandelate":"urinary_antiseptic",
               "fidaxomicin":"other"}
STRIP_TOKENS = {"dextrs","dextrose","hcl","hci","nacl","hyclate","monohydrate",
                "macrocrystal","axetil","tromethamine","sodium_citrate","pot",
                "in","sodium_chloride","d5w","ns","bacteriostatic","pack",
                "phosphate","ethylsuccinate","dexamethasone","hippurate",
                "mandelate","tromethamin","citrate","sulfate","sodium",
                "diluent","combo","chloride","iso","sulfa"}
ALIAS = {"zithromax":"azithromycin","bactrim_ds":"trimethoprim_sulfamethoxazole",
         "bactrim":"trimethoprim_sulfamethoxazole","cipro":"ciprofloxacin",
         "augmentin":"amoxicillin_clavulanate","keflex":"cephalexin",
         "macrobid":"nitrofurantoin","macrodantin":"nitrofurantoin",
         "levaquin":"levofloxacin","xifaxan":"rifaximin","flagyl":"metronidazole",
         "hiprex":"methenamine","zyvox":"linezolid",
         "sulfamethoxazole_trimethoprim":"trimethoprim_sulfamethoxazole"}
DRUG_CLASS.update(EXTRA_DRUGS)

import re
def resolve_drug(name):
    d = norm(pd.Series([name]))[0]
    if d in DRUG_CLASS: return d
    if d in ALIAS: return ALIAS[d]
    toks = [t for t in re.split(r"[_\s]+", d) if t and t not in STRIP_TOKENS]
    d2 = "_".join(toks)
    if d2 in DRUG_CLASS: return d2
    if d2 in ALIAS: return ALIAS[d2]
    return None

def med_vocab(site, path, tag):
    c = Counter()
    for chunk in pd.read_csv(path, usecols=["medication_name"], chunksize=2000000,
                             low_memory=False, encoding="utf-8-sig"):
        for v, k in chunk["medication_name"].astype(str).value_counts().items():
            c[v] += k
    return c

s_med = med_vocab(S, os.path.join(S, "microbiology_cultures_prior_med.csv"), "S")
u_med = med_vocab(U, os.path.join(U, "microbiology_cultures_prior_med.csv"), "U")
p(f"S medication_name 取值: {len(s_med):,} | U: {len(u_med):,}")

def match_report(med, tag):
    hit = 0; tot = 0; miss = []
    for v, k in med.items():
        d = resolve_drug(v)
        tot += k
        if d is not None:
            hit += k
        else:
            miss.append((v, k))
    p(f"[{tag}] 药物名匹配覆盖(剥离后): {hit:,}/{tot:,} = {hit/tot:.4f}")
    miss.sort(key=lambda x: -x[1])
    p(f"[{tag}] 仍未匹配 top15: {miss[:15]}")
    return miss

miss_s = match_report(s_med, "S")
miss_u = match_report(u_med, "U")

# 补充药并入规范字典导出
for d, cls in EXTRA_DRUGS.items():
    rows.append(dict(drug=d, drug_display=d.upper(), canonical_class=cls,
                     antipseudomonal=int(d in ANTIPSEUDO), count=0))
pd.DataFrame([dict(drug=d, drug_display=r["drug_display"], canonical_class=r["canonical_class"],
                   antipseudomonal=r["antipseudomonal"], count=r["count"])
              for d, r in [(x["drug"], x) for x in rows]]).to_csv(
    os.path.join(DICTS, "dict_canonical_drugs.csv"), index=False)
p("-> dict_canonical_drugs.csv 已存")

# ============ 2. 属级映射 ============
log("2. 属级映射")
po = pd.read_csv(os.path.join(VOCAB, "vocab_MGB_prior_org.csv"))
GENUS_EXC = {"STREPTOCOCCUS MITIS/ORALIS GROUP": "STREPTOCOCCUS"}
def genus_of(v):
    if v in GENUS_EXC: return GENUS_EXC[v]
    return v.split()[0]
genus_rows = [dict(prior_org=v, genus=genus_of(v), count=int(c)) for v, c in
              zip(po["value"], po["count"])]
pd.DataFrame(genus_rows).to_csv(os.path.join(DICTS, "dict_genus_map.csv"), index=False)
p(f"属级映射 {len(genus_rows)} 行 -> dict_genus_map.csv (PSEUDOMONAS 行: "
  f"{sum(1 for r in genus_rows if 'PSEUDOMONAS' in r['prior_org'])})")

# ============ 3. 合并症词汇交集 ============
log("3. 合并症词汇交集")
mgb_c = set(pd.read_csv(os.path.join(VOCAB, "vocab_MGB_comorbidity_category.csv"))["value"].astype(str))
s_c = set(pd.read_csv(os.path.join(VOCAB, "vocab_S_comorbidity_component.csv"))["value"].astype(str))
u_c = set(pd.read_csv(os.path.join(VOCAB, "vocab_U_comorbidity_component.csv"))["value"].astype(str))
inter_su = s_c & u_c
inter_all = mgb_c & inter_su
p(f"MGB {len(mgb_c)} | S {len(s_c)} | U {len(u_c)}")
p(f"S∩U: {len(inter_su)} | MGB∩S∩U: {len(inter_all)}")
p(f"MGB∩S: {len(mgb_c & s_c)} | MGB∩U: {len(mgb_c & u_c)}")
p("\n三站共有样例(10):", sorted(list(inter_all))[:10])
p("\nS/U 共有但 MGB 无(10):", sorted(inter_su - mgb_c)[:10])
p("\nMGB 独有(10):", sorted(mgb_c - inter_su)[:10])
p("\n映射策略: 三站共有的直接用共有标签; MGB独有→映射到 S/U 最近类别(人工审核); "
  "差异清单另存")
pd.DataFrame([dict(value=v, in_mgb=v in mgb_c, in_s=v in s_c, in_u=v in u_c)
              for v in sorted(mgb_c | s_c | u_c)]).to_csv(
    os.path.join(DICTS, "dict_comorbidity_intersection.csv"), index=False)
p("-> dict_comorbidity_intersection.csv 已存")

# ============ 4. 年龄分箱 ============
log("4. 年龄分箱")
bins = {}
for tag, path in [("MGB", os.path.join(VOCAB, "vocab_MGB_age_bins.csv")),
                  ("S", os.path.join(VOCAB, "vocab_S_age_bins.csv")),
                  ("U", os.path.join(VOCAB, "vocab_U_age_bins.csv"))]:
    df = pd.read_csv(path)
    bins[tag] = set(df["value"].astype(str).str.replace(" years", "", regex=False))
    p(f"[{tag}] {len(bins[tag])} 箱: {sorted(bins[tag])}")
same = bins["MGB"] == bins["S"] == bins["U"]
p(f"三站分箱一致: {same}")
if same:
    bins_sorted = sorted(bins["MGB"])
    pd.DataFrame([dict(unified_bin=b) for b in bins_sorted]).to_csv(
        os.path.join(DICTS, "dict_age_bins.csv"), index=False)
    p("-> dict_age_bins.csv 已存 (去后缀即统一)")

log("完成")
f.close()
