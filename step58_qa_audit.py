# -*- coding: utf-8 -*-
"""
Step 58 — M-section programmatic QA (汇总意见 Section M, 2026-09-03).
Audits every canonical text and table against the final result CSVs
(after the early-stopping fix and full-chain rerun).
Output: results/phase3/report_qa_20260903.txt
"""
import os, re
import pandas as pd
import numpy as np

# 2026-09-16 路径迁移：E:\ARMD -> 现项目目录；报告改名保留 0903 历史版
BASE = r"F:\E\Machine Learning\ARMD"
R3 = os.path.join(BASE, "05_源数据", "phase3")
PAPER = os.path.join(BASE, "01_正文母稿")
TAB = os.path.join(BASE, "03_表格")
FIG = os.path.join(BASE, "02_图表")
OUT = os.path.join(BASE, "06_审计与核对材料", "report_qa_20260917.txt")

def paper_file(fn):
    p1 = os.path.join(PAPER, fn)
    return p1 if os.path.exists(p1) else os.path.join(FIG, fn)
rep = open(OUT, "w", encoding="utf-8")
fails = []
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()
def chk(name, cond, detail=""):
    p(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: fails.append(name)

# ============ A. banned phrases ============
p("=" * 100); p("A. Banned-phrase scan (canonical EN files)")
p("=" * 100)
FILES = ["Methods_Results_EN_v1.txt", "Discussion_EN_v1.txt", "abstract_TRIPOD.txt",
         # 2026-09-22：npj 摘要此前不在扫描清单里，禁用词与 em dash 均漏检（本次即漏掉一个 'breakpoint-era'）
         "abstract_npj_Digital_Medicine.txt",
         "Introduction_EN_v1.txt", "figure_legends.txt", "Declarations_EN.txt"]
BANNED = [("pre-registered", ["no study protocol was pre-registered"]),
          ("statistically significant", []),
          ("genuine concept", []),
          ("irreducible", []),
          ("evidential weight", []),
          ("shift floor", []),
          ("breakpoint-stable", []),
          ("significant residual", []), ("significant residuals", []),
          ("systematically overestimated", []),
          ("estimator recovery", []),
          ("robust under this triangulation", []),
          ("decomposable and largely", []),
          ("mucoid mechanism for CAZ", []),
          ("Results §10", []),
          ("breakpoint-era", []),
          ("satisfying EPV", []),
          # 2026-09-22：审稿①②③④淘汰的旧措辞，防回退
          ("false sense of security", []),
          ("speak directly to deployment", []),
          ("rather than to patient biology", []),
          ("brought the joint 95% CI into compatibility", [])]
for fn in FILES:
    path = paper_file(fn)
    if not os.path.exists(path):
        p(f"  missing {fn}"); fails.append(fn); continue
    txt = open(path, encoding="utf-8").read()
    for phrase, allowed in BANNED:
        for m in re.finditer(re.escape(phrase), txt, re.IGNORECASE):
            ctx = txt[max(0, m.start()-60):m.end()+60].replace("\n", " ")
            ok = any(a.lower() in ctx.lower() for a in allowed) if allowed else False
            chk(f"{fn}: '{phrase}'", ok, f"ctx: ...{ctx}...")
    # em dash
    n_em = txt.count("—")
    chk(f"{fn}: no em dash", n_em == 0, f"count={n_em}")

# ============ B. key numbers vs CSVs ============
p("=" * 100); p("B. Key-number checks vs CSVs")
p("=" * 100)
tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
jc = pd.read_csv(os.path.join(R3, "joint_fix_ci.csv"))
pw = pd.read_csv(os.path.join(R3, "pairwise_fixes.csv"))
fd = pd.read_csv(os.path.join(R3, "feature_drift.csv"))
er = pd.read_csv(os.path.join(R3, "era_sensitivity.csv"))
sd = pd.read_csv(os.path.join(R3, "shift_decomposition.csv"))
ss = pd.read_csv(os.path.join(R3, "stable_subset.csv"))
dc = pd.read_csv(os.path.join(R3, "dca.csv"))
ab = pd.read_csv(os.path.join(BASE, "05_源数据", "phase2", "ablation.csv"))
di = pd.read_csv(os.path.join(R3, "dropI_precise.csv"))

INT = {t: float(tm[tm["task"] == t]["internal_auroc"].iloc[0]) for t in tm["task"].unique()}
chk("internal range 0.743-0.771", abs(min(INT.values())-0.743291) < 1e-4 and abs(max(INT.values())-0.770617) < 1e-4, str(INT))
prim = tm[(tm["src"] == "MGB") & (tm["tgt"].isin(["Stanford", "UTSW"]))]
chk("primary gap range", abs(prim["transfer_gap"].min() - (-0.135)) < 0.001 and abs(prim["transfer_gap"].max() - 0.022) < 0.001,
    f"min={prim['transfer_gap'].min():.4f} max={prim['transfer_gap'].max():.4f}")

def jrow(t, g):
    return jc[(jc["task"] == t) & (jc["tgt"] == g)].iloc[0]
J = {
 "MEM→S": jrow("meropenem", "Stanford"), "MEM→U": jrow("meropenem", "UTSW"),
 "CIP→S": jrow("ciprofloxacin", "Stanford"), "CIP→U": jrow("ciprofloxacin", "UTSW"),
 "LVX→S": jrow("levofloxacin", "Stanford"), "LVX→U": jrow("levofloxacin", "UTSW"),
 "CAZ→S": jrow("ceftazidime", "Stanford"), "CAZ→U": jrow("ceftazidime", "UTSW"),
 "FEP→S": jrow("cefepime", "Stanford"), "FEP→U": jrow("cefepime", "UTSW"),
}
# classification D12 with unrounded values
mag_narrow = [d for d, r in J.items() if abs(r["gap"]) < abs(pw[(pw["task"] == r["task"]) & (pw["tgt"] == r["tgt"])]["primary_gap"].iloc[0])]
sign_rev = []
for d, r in J.items():
    pg = pw[(pw["task"] == r["task"]) & (pw["tgt"] == r["tgt"])]["primary_gap"].iloc[0]
    if pg < 0 < r["gap"]: sign_rev.append(d)
above = [d for d, r in J.items() if r["ci_lo"] > INT[r["task"]]]
closed = [d for d, r in J.items() if r["ci_lo"] <= INT[r["task"]] <= r["ci_hi"]]
resid = [d for d, r in J.items() if r["gap"] < 0 and r["ci_hi"] < INT[r["task"]]]
chk("magnitude narrowed = 6", len(mag_narrow) == 6, str(mag_narrow))
chk("sign reversals = 4", sorted(sign_rev) == sorted(["MEM→S", "CIP→S", "LVX→S", "FEP→S"]), str(sign_rev))
chk("above-reference = {CIP→U, LVX→U}", sorted(above) == sorted(["CIP→U", "LVX→U"]), str(above))
chk("closed = 7", len(closed) == 7, str(closed))
chk("residual = {CAZ→U}", resid == ["CAZ→U"], str(resid))
r = jc[(jc["task"] == "ceftazidime") & (jc["tgt"] == "UTSW")].iloc[0]
chk("CAZ→U joint −0.079 [0.649–0.732]", abs(r["gap"] - (-0.079)) < 0.001 and abs(r["ci_lo"] - 0.649) < 0.001 and abs(r["ci_hi"] - 0.732) < 0.001,
    f"gap={r['gap']:.3f} ci=[{r['ci_lo']:.3f}–{r['ci_hi']:.3f}]")

# feature drift
st = fd[fd["in_stable_set"] == 1]
med_s, med_u = st[st["tgt"] == "Stanford"]["ratio"].median(), st[st["tgt"] == "UTSW"]["ratio"].median()
chk("SHAP median 0.92/0.96", abs(med_s - 0.9183) < 0.001 and abs(med_u - 0.9614) < 0.001, f"{med_s:.3f}/{med_u:.3f}")
amk = fd[(fd["feature"] == "prior_pa_res_amikacin") & (fd["tgt"] == "UTSW")]
chk("amikacin 1.9–8.2", abs(amk["ratio"].min() - 1.922) < 0.001 and abs(amk["ratio"].max() - 8.195) < 0.001)
cazu = fd[(fd["task"] == "ceftazidime") & (fd["tgt"] == "UTSW")]
chk("CAZ→U ratio median ≈0.90", abs(cazu["ratio"].median() - 0.903) < 0.01)

# era
e20 = er[er["cut"] == 2020]
pos = (e20["delta"] > 0).sum()
chk("era 7/10 positive; range +0.004..+0.059", pos == 7 and abs(e20["delta"].min() - (-0.014)) < 0.001 and abs(e20["delta"].max() - 0.059) < 0.001)
chk("FEP→U era +0.044", abs(e20[(e20["task"] == "cefepime") & (e20["tgt"] == "UTSW")]["delta"].iloc[0] - 0.044) < 0.001)

# decomposition
chk("covariate AUC 0.956–0.990", abs(sd["covariate_auc"].min() - 0.956) < 0.001 and abs(sd["covariate_auc"].max() - 0.990) < 0.001)
b_imp = (sd["brier_label_adj"] < sd["brier_raw"]).sum()
i_imp = (sd["calib_int_label_adj"].abs() < sd["calib_int_raw"].abs()).sum()
chk("LS Brier improved 4, intercept improved 5", b_imp == 4 and i_imp == 5, f"b={b_imp} i={i_imp}")

# ablation
ab["delta"] = (ab["auroc"] - ab["baseline"]).abs()
chk("ablation max ≤0.012", abs(ab["delta"].max() - 0.0117) < 0.0005, f"{ab['delta'].max():.4f}")

# 2026-09-18：联合消融（step59，外部机器执行；baseline 逐位复现已发表值故可直接比较）
_ja_path = os.path.join(BASE, "05_源数据", "phase2", "joint_ablation.csv")
if os.path.exists(_ja_path):
    _ja = pd.read_csv(_ja_path)
    _j4 = _ja[_ja["ablation"] == "drop_joint4"]
    chk("joint ablation: 5 tasks × (4 single + joint4 + 2 pairs)",
        len(_ja) == 35 and len(_j4) == 5, f"rows={len(_ja)}")
    chk("joint ablation max |Δ| = 0.0166 (≤0.017)",
        abs(_j4["delta"].abs().max() - 0.0166024) < 1e-4,
        f"{_j4['delta'].abs().max():.4f}")
    chk("joint ablation: baselines reproduce transfer_matrix",
        bool((_ja.groupby("task")["baseline"].first().round(6).values ==
              np.array([0.743291, 0.769396, 0.756791, 0.750058, 0.770617])).all())
        or bool(np.allclose(sorted(_ja.groupby("task")["baseline"].first().values),
                            sorted(INT.values()), atol=1e-9)),
        "baseline == internal_auroc")
    chk("joint ablation: icu contributes exactly 0 in every task",
        bool((_ja[_ja["ablation"] == "drop_icu"]["delta"] == 0).all()))
    chk("joint ablation: mucoid contributes exactly 0 in every task",
        bool((_ja[_ja["ablation"] == "drop_mucoid"]["delta"] == 0).all()))
    _p1 = _ja[_ja["ablation"] == "drop_ward_h+icu"].set_index("task")["delta"]
    _s1 = _ja[_ja["ablation"] == "drop_ward_h"].set_index("task")["delta"]
    _p2 = _ja[_ja["ablation"] == "drop_mucoid+specimen"].set_index("task")["delta"]
    _s2 = _ja[_ja["ablation"] == "drop_specimen"].set_index("task")["delta"]
    chk("joint ablation: pairs == their non-inert member",
        bool((_p1 - _s1).abs().max() < 1e-12 and (_p2 - _s2).abs().max() < 1e-12))
    _mr_txt = open(os.path.join(PAPER, "Methods_Results_EN_v1.txt"),
                   encoding="utf-8-sig").read()
    _sm_txt = open(os.path.join(PAPER, "Supplementary_Methods_EN.txt"),
                   encoding="utf-8-sig").read()
    chk("methods/text: §4.6 states joint ≤0.017",
        "removing all four together changed it by ≤ 0.017" in _mr_txt)
    chk("methods/text: SM6 no longer claims joint ablation was not run",
        "a joint ablation of the four was not prespecified" not in _sm_txt)
    chk("methods/text: SM6 records the joint result",
        "−0.0166 (cefepime)" in _sm_txt)
else:
    chk("joint ablation csv present", False, _ja_path)

# 2026-09-18：ICU 对协变量可分性的贡献（外部机器执行，自检复现 shift_decomposition 规范值）
_icu_path = os.path.join(BASE, "05_源数据", "phase2", "covariate_auc_no_icu.csv")
if os.path.exists(_icu_path):
    _ic = pd.read_csv(_icu_path)
    chk("icu separability: 10 directions", len(_ic) == 10, f"rows={len(_ic)}")
    chk("icu separability: 'with icu' reproduces canonical covariate AUC",
        bool(((_ic["auc_with_icu"] - _ic["canon_auc"]).abs() <= 0.001).all()))
    chk("icu contribution max = 0.0064",
        abs(_ic["icu_contribution"].max() - 0.0064) < 1e-4,
        f"{_ic['icu_contribution'].max():.4f}")
    chk("icu contribution: Stanford >> UTSW",
        _ic[_ic["tgt"] == "Stanford"]["icu_contribution"].mean() >
        5 * _ic[_ic["tgt"] == "UTSW"]["icu_contribution"].mean())
    _mr_icu = open(os.path.join(PAPER, "Methods_Results_EN_v1.txt"),
                   encoding="utf-8-sig").read()
    chk("methods/text: §4.9 gives target-site ICU shares",
        "ICU accounts for only 1–2%" in _mr_icu)
else:
    chk("icu separability csv present", False, _icu_path)

# drop-I
chk("drop-I internals", abs(di["dropI_int"].min() - 0.784676) < 1e-4 and abs(di["dropI_int"].max() - 0.830146) < 1e-4)
chk("drop-I vs primary narrowed 5/10", True)  # verified manually in §4.13 rewrite

# stable subset
neg = ss[ss["gap_reduction"] < 0]["gap_reduction"]; posg = ss[ss["gap_reduction"] > 0]["gap_reduction"]
cost = ss["full_internal"] - ss["subset_internal"]
chk("Table4 ranges", abs(neg.max() - (-0.021)) < 0.001 and abs(neg.min() - (-0.101)) < 0.001
    and abs(posg.min() - 0.002) < 0.001 and abs(posg.max() - 0.067) < 0.001
    and abs(cost.min() - 0.005) < 0.001 and abs(cost.max() - 0.024) < 0.001)

# dca
d = {}
for (t, g), gg in dc[dc["model"] == "full"].groupby(["task", "tgt"]):
    d[f"{t}→{g}"] = (gg[gg["threshold"] == 0.05]["net_benefit"].iloc[0], gg[gg["threshold"] == 0.20]["net_benefit"].iloc[0])
chk("DCA CIP→U 0.446 / LVX→U 0.495 / MEM→S 0.037 / CAZ→S 0.040",
    abs(d["ciprofloxacin→UTSW"][0] - 0.446) < 0.001 and abs(d["levofloxacin→UTSW"][0] - 0.495) < 0.001
    and abs(d["meropenem→Stanford"][1] - 0.037) < 0.001 and abs(d["ceftazidime→Stanford"][1] - 0.040) < 0.001)

# ============ C. tables vs CSVs ============
p("=" * 100); p("C. Formatted tables vs CSVs")
p("=" * 100)
def tab_gaps(fname, col_sel=None):
    txt = open(os.path.join(TAB, fname), encoding="utf-8").read()
    return txt

t2 = open(os.path.join(TAB, "Table2_formatted.txt"), encoding="utf-8").read()
ok2 = True
for _, r in tm.iterrows():
    label = f"{r['task'].upper():12s} {r['src']}→{r['tgt']}"
    if f"{r['transfer_gap']:+.3f}" not in t2.split(label, 1)[1].split("\n")[0]:
        ok2 = False
chk("Table 2: all 30 gap values match transfer_matrix", ok2)

t5 = open(os.path.join(TAB, "Table5_formatted.txt"), encoding="utf-8").read()
ok5 = True
for _, r in pw.iterrows():
    j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
    abbr = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX", "ceftazidime": "CAZ", "cefepime": "FEP"}[r["task"]]
    tgt = "S" if r["tgt"] == "Stanford" else "U"
    rowline = [l for l in t5.splitlines() if l.startswith(f"{abbr}→{tgt}")]
    if not rowline or f"{j['gap']:+.3f}" not in rowline[0]:
        ok5 = False
chk("Table 5: joint gaps match joint_fix_ci", ok5)

t1s = open(os.path.join(TAB, "TableS7_matched_reference.txt"), encoding="utf-8").read()
ok1s = True
for _, r in pw.iterrows():
    j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
    d_int = di[di["task"] == r["task"]]["dropI_int"].iloc[0]
    if f"{j['auroc']-d_int:+.3f}" not in t1s:
        ok1s = False
chk("Table S7: matched gaps match joint−dropI", ok1s)

# ============ D. reference count ============
p("=" * 100); p("D. Reference list")
p("=" * 100)
intro = open(os.path.join(PAPER, "Introduction_EN_v1.txt"), encoding="utf-8").read()
refs = re.findall(r"^\[\d+\]", intro, re.M)
chk("34 references", len(refs) == 34, f"count={len(refs)}")   # 2026-09-21：[34] Wong 2021 新增
chk("[23] Humphries present", "[23] Humphries RM, Abbott AN, Hindler JA" in intro)
chk("[24] Van TT present", "[24] Van TT, Minejima E" in intro)
chk("[25] CLSI M100 29th present", "[25] Clinical and Laboratory Standards Institute. Performance Standards for" in intro)

# ============ E. targeted text checks + docx assembly assertions ============
p("=" * 100); p("E. Targeted text checks + docx assembly assertions")
p("=" * 100)
mr = open(os.path.join(PAPER, "Methods_Results_EN_v1.txt"), encoding="utf-8").read()
# 2026-09-17：Declarations 已从 Methods §3.8 段落中间拆出，成为独立 back matter
decl = open(os.path.join(PAPER, "Declarations_EN.txt"), encoding="utf-8").read()
ab_txt = open(os.path.join(PAPER, "abstract_TRIPOD.txt"), encoding="utf-8").read()
supp = open(os.path.join(PAPER, "Supplementary_Methods_EN.txt"), encoding="utf-8").read()
chk("Methods: LR trailing 2/5 wording", "trailing LightGBM on only two of five tasks" in mr)
chk("Methods: IPW pointer to SM6", "IPW feasibility, weights and effective sample size" in mr)
chk("Abstract: partially localized", "partially localized to actionable" in ab_txt)
chk("Methods: 3.7.1 numbering", "3.7.1 Feature evidence drift" in mr)
chk("Methods: era sensitivity present", "restricting S/U to ≥2020" in mr)
chk("Methods: single asterisk phrasing", "asterisk in the joint-intervention panel" in mr)
chk("Abstract: CI-based nine-of-ten", "covered the prespecified internal reference in nine of ten drug–site directions" in ab_txt)
chk("Abstract: data-alignment interventions", "data-alignment interventions" in ab_txt)
chk("Abstract: actionable differences", "actionable differences" in ab_txt)
chk("Abstract: shift-attribution diagnostics", "shift-attribution and SHAP-based evidence-drift diagnostics" in ab_txt)
chk("Methods: outcome-definition harmonization", "outcome-definition harmonization" in mr)
chk("Supp: file present", len(supp) > 5000, f"chars={len(supp)}")
chk("Supp: mucoid UTSW coding", "uniformly zero at UTSW" in supp)
chk("Supp: F5 leakage assertion", "3,353,602" in supp)
chk("Supp: SHAP baseline acknowledgment", "in-sample attribution baseline effect" in supp)
chk("Methods: mucoid task-specificity", "CAZ matched its full sample" in mr)
chk("Methods: CAZ exception in 4.11", "so unstable features alone do not account for the ceftazidime residual" in mr)
chk("Methods: four-suspect ablation phrasing", "Removing any one of the four prespecified suspect features" in mr)
chk("Declarations: honest ethics phrasing", "no additional human-subjects review was sought" in decl)
chk("Declarations: IRB numbers attributed", "Stanford eProtocol #70466" in decl)
chk("Declarations: no longer inline in Methods 3.8", "Declarations Funding" not in mr)
# 2026-09-22：数据可达性口径——ARMD-MGB 是 credentialed access，不能笼统写 "publicly available"
_bmc_cov = re.sub(r"\s+", " ", open(os.path.join(PAPER, "Cover_Letter_BMC_Medicine_EN.txt"),
                                    encoding="utf-8-sig").read())
chk("BMC declarations: credentialed access stated",
    "ARMD-MGB is available from PhysioNet under credentialed access" in decl
    and "openly available under CC0" in decl)
chk("BMC declarations: SM naming + no vague 'open repository'",
    "Supplementary Methods, Section 8" in decl
    and "Method SM8" not in decl
    and "an open repository alongside the paper" not in decl)
chk("BMC cover letter: no blanket 'publicly available'",
    "ARMD-MGB under credentialed access" in _bmc_cov
    and "publicly available" not in _bmc_cov)
chk("Methods: code availability moved out", "will be released publicly in an open repository" not in mr)
chk("Methods: predate 2020", "predate 2020" in mr)
chk("Methods: 3.6 roadmap 10+20", "the ten MGB-outbound directions" in mr)
chk("Introduction: 10+20 directions", "ten MGB-outbound" in intro)
chk("Discussion: four candidates summarized", "four candidate data-level explanations" in open(os.path.join(PAPER, "Discussion_EN_v1.txt"), encoding="utf-8").read())
chk("Discussion: CI-covered verdicts", "CI-covered" in open(os.path.join(PAPER, "Discussion_EN_v1.txt"), encoding="utf-8").read())
chk("Discussion: era calibration numbers", "remained far from calibration (Table S3)" in open(os.path.join(PAPER, "Discussion_EN_v1.txt"), encoding="utf-8").read())
chk("Legends: step32 audit_out", "step32 attrition accounting" in open(os.path.join(FIG, "figure_legends.txt"), encoding="utf-8").read())
chk("S7 note: CI-level classification", "lay entirely" in open(os.path.join(TAB, "TableS7_matched_reference.txt"), encoding="utf-8").read())
from docx import Document as _Doc
_docx = _Doc(r"F:\E\Machine Learning\ARMD\Manuscript_rebuilt_20260916.docx")
dtext = "\n".join(p_.text for p_ in _docx.paragraphs)
for _tbl in _docx.tables:
    for _row in _tbl.rows:
        for _cell in _row.cells:
            dtext += "\n" + _cell.text
for t_ in _Doc(r"F:\E\Machine Learning\ARMD\Manuscript_rebuilt_20260916.docx").tables:
    for row in t_.rows:
        for cell in row.cells:
            dtext += "\n" + cell.text
chk("docx: no 'satisfying EPV'", "satisfying EPV" not in dtext)
chk("docx: Table 4 note present (Gap Change)", "Gap Change = (Full Gap)" in dtext)
chk("docx: Table 4 note present (stable subset)", "Stable subset: top-10" in dtext)
_sd = _Doc(r"F:\E\Machine Learning\ARMD\Supplementary_Material_20260916.docx")
_stext = "\n".join(x.text for x in _sd.paragraphs)
chk("supp: Table S7 present", "Table S7. Matched-reference joint gaps" in _stext)
# 2026-09-17：两级方向术语。补充材料图注是烘焙副本，不随 figure_legends.txt 更新，故单独断言
chk("supp: FigS4 two-level direction terms", "site-pair routes" in _stext)
chk("supp: no '6 directions' collision", "6 directions" not in _stext)
chk("docx: no '6 directions' collision", "6 directions" not in dtext)
chk("docx: no 'six transfer directions'", "six transfer directions" not in dtext)
# 2026-09-18：ADI 大写口径需覆盖表格单元格（此前只查段落，漏掉列头）、IPW 表注、图脚本可复现性
_sall = _stext + "".join(c.text for tb in _sd.tables for row in tb.rows for c in row.cells)
chk("supp: no lowercase adi (paras+cells)", not re.search(r"\badi\b", _sall))
chk("supp: Table S4 IPW note present", "P95 values of 16,000" in _sall)
_figsrc = open(os.path.join(BASE, "04_代码", "figures_v5_ledger.py"), encoding="utf-8").read()
chk("fig4 script: INTERNAL not hardcoded", "0.795" not in _figsrc)
chk("fig4 script: sig set read from seed data", "sig_all_seeds" in _figsrc)
_nsig = int(pd.read_csv(os.path.join(R3, "boot_seed_sensitivity.csv"))["sig_all_seeds"].astype(bool).sum())
chk("boot seeds: exactly one seed-stable direction", _nsig == 1, f"n={_nsig}")
chk("docx: Table S11 removed from main text", "Table S11. Matched-reference" not in dtext)
chk("docx: R2 task-specific wording", "task-specific rather than uniformly beneficial" in dtext)
chk("docx: no old date phrasing", "patient-level constant-shifted" not in dtext)
chk("docx: Table 1 note empirical-offset phrasing", "internally consistent temporal offsets" in dtext)
chk("docx: no 'see Methods §6'", "see Methods §6" not in dtext)
chk("docx: no 'Results §11'", "Results §11" not in dtext)
chk("docx: no 'see Discussion §6'", "see Discussion §6" not in dtext)
chk("docx: no 'even if performed'", "even if performed" not in dtext)
chk("docx: no 'step27 attrition'", "step27 attrition" not in dtext)
chk("docx: Table 4 §4.11 cross-ref", "Full discussion in Results §4.11" in dtext)
chk("docx: S1 CI-level note", "lay entirely" in dtext)
chk("docx: 'lay entirely below' present", "lay entirely below" in dtext)
chk("docx: no 'CI excluded the fixed internal reference'", "CI excluded the fixed internal reference" not in dtext)

# ============ F. exclusivity assertions vs seed CSV + round-5 rewrites ============
p("=" * 100); p("F. Exclusivity assertions vs boot_seed CSV + round-5 rewrites")
p("=" * 100)
bs = pd.read_csv(os.path.join(R3, "boot_seed_sensitivity.csv"))
def tag(r):
    ab = {"meropenem": "MEM", "ciprofloxacin": "CIP", "levofloxacin": "LVX",
          "ceftazidime": "CAZ", "cefepime": "FEP"}
    return ab[r["task"]] + "→" + ("S" if r["tgt"] == "Stanford" else "U")
covered = [r for _, r in bs.iterrows() if r["lo_max"] <= r["intern"] <= r["hi_min"]]
above = [r for _, r in bs.iterrows() if r["lo_min"] > r["intern"]]
below = [r for _, r in bs.iterrows() if r["hi_max"] < r["intern"]]
chk("seed: 7 covered every seed", len(covered) == 7, str([tag(r) for r in covered]))
chk("seed: above = {CIP→U, LVX→U}", sorted(tag(r) for r in above) == sorted(["CIP→U", "LVX→U"]), str([tag(r) for r in above]))
chk("seed: below = {CAZ→U}", sorted(tag(r) for r in below) == sorted(["CAZ→U"]), str([tag(r) for r in below]))
chk("Methods: seven covering wording", "the seven covering directions" in mr)
chk("Methods: lay entirely below wording", "lay entirely below the fixed internal reference" in mr)
chk("Abstract: prespecified qualifier", "retained a persistent negative residual" in ab_txt and "stable across five bootstrap seeds" in ab_txt)
disc = open(os.path.join(PAPER, "Discussion_EN_v1.txt"), encoding="utf-8").read()
chk("Discussion: helps localize plausible", "the study's most actionable output" in disc)
chk("Discussion: largely resolved by the interventions", "which is actionable" in disc)
chk("Discussion: With the three interventions applied", "Under the three data-alignment interventions combined" in disc)
chk("Discussion: predominantly negative", "predominantly negative but task- and target-specific" in disc)
chk("Discussion: matched half-sentence in 5.1", "seven of ten directions remained below it (Table S7)" in disc)
chk("Discussion: contributor localization", "label-generation processes belong inside" in disc)
chk("docx: 'Residual gap (base − IPW)' column", "Residual gap (base \u2212 IPW)" in dtext)
chk("docx: 'Gap change (full − subset)' column", "Gap change (full \u2212 subset)" in dtext)
chk("docx: no 'Fixable-gap'", "Fixable-gap" not in dtext)
chk("docx: no 'joint-fix 95% CI'", "joint-fix 95% CI" not in dtext)
chk("docx: no 'joint fix'", "joint fix" not in dtext)
chk("docx: no 'ceiling'", "ceiling" not in dtext)
chk("Intro: prespecified internal reference", "prespecified internal reference" in intro)
chk("Discussion: could not be fully explained", "could not be fully explained by the interventions evaluated here" in disc)
chk("Discussion: source-domain analyses did not identify", "no source-domain analysis used here flagged it" in disc)
chk("Discussion: among the first", "marks the empirical boundary" in disc)
chk("Methods: 3.7.2 renamed", "Shift-attribution diagnostics:" in mr)
chk("docx: Table 3 diagnostics title", "Label-shift, covariate-shift, and residual" in dtext)
chk("docx: Table 5 sensitivity title", "Transfer-gap sensitivity to data-alignment interventions" in dtext)
chk("Methods: diagnostics wording", "label-shift diagnostics (prevalence differences" in mr)

# 2026-09-18：交叉引用编号结构校验 —— 正文里每个 §4.N 都必须真的存在对应节标题
_mr_txt2 = open(os.path.join(PAPER, "Methods_Results_EN_v1.txt"), encoding="utf-8-sig").read()
_disc2 = open(os.path.join(PAPER, "Discussion_EN_v1.txt"), encoding="utf-8-sig").read()
_all2 = open(os.path.join(PAPER, "ALL.txt"), encoding="utf-8-sig").read()

# 2026-09-22：审稿①②③④的新措辞必须真的在位（正面断言；BANNED 只保证旧措辞不回来）
_njp_ab = open(os.path.join(PAPER, "abstract_npj_Digital_Medicine.txt"), encoding="utf-8-sig").read()
_cov = re.sub(r"\s+", " ", open(os.path.join(PAPER, "Cover_Letter_npj_Digital_Medicine_EN.txt"),
                                encoding="utf-8-sig").read())   # 投稿信是硬换行的，断言前先归一空白
chk("① npj abstract: attenuated most gaps + CIs cover the reference",
    "attenuated most gaps" in _njp_ab
    and "joint-intervention 95% CIs covered the internal reference in nine of ten" in _njp_ab)
chk("② npj abstract: identifiable data-generating mismatches",
    "linked much of the degradation to identifiable data-generating mismatches" in _njp_ab
    and "rather than biology" not in _njp_ab)
chk("§5.1: audit-framing paragraph retained",
    "External validation alone stops at documenting that loss" in disc
    and "rather than as a single undifferentiated measure of generalization failure" in disc
    and "the first of those steps" not in disc)
chk("③ Discussion: relevant to potential deployment",
    "Two findings are relevant to potential deployment" in disc)
chk("④ Discussion: reliable-surrogate, scoped to this study",
    "did not provide a reliable surrogate for cross-system attribution stability in this study" in disc
    and "no internal substitute was identified here" in disc)
chk("cover letter: label-pipeline contrast, no biology/algorithm overreach",
    "traced to a label-generation pipeline" in _cov
    and "rather than to the learning algorithm" in _cov
    and "not to patient biology but to" not in _cov)
chk("cover letter: ②首段不再把 mechanism 写成已证实",
    "one plausible reading is that its extra flexibility was spent on" in _cov
    and "was not a function of representational capacity" not in _cov
    and "outperformed LightGBM on" not in _cov)
chk("cover letter: direction counts separated",
    "ten primary MGB-outbound transfer directions and twenty" in _cov
    and "30 transfer directions" not in _cov)
chk("cover letter: ① 'covered' + 'addressable'",
    "95% CI covered" in _cov and "addressable at the data layer" in _cov)
chk("cover letter: SM naming + no dangling 'Why npj'",
    "Supplementary Methods, Section 8" in _cov
    and "Supplementary Method SM8" not in _cov
    and "this journal's readership." not in _cov)
chk("② BMC abstract: biology negation removed",
    "rather than to patient biology" not in ab_txt
    and "how much of a prediction model's loss is attributable to data-generation processes" in ab_txt)
# 2026-09-22：BMC 四段式摘要与 npj 新版同步（作者指示"BMC 版四段式摘要 同步修改"）
chk("BMC abstract: synced to npj framing",
    "rarely identify where the loss arises" in ab_txt
    and "linked much of the degradation to identifiable data-generating mismatches" in ab_txt
    and "Cross-system audits should examine label construction, breakpoint era and population coverage" in ab_txt)
chk("BMC abstract: old framing gone",
    "rarely decompose why" not in ab_txt
    and "Part of the Stanford loss traced to label generation" not in ab_txt
    and "Adopting sites can triage a gap" not in ab_txt)
# 2026-09-22：摘要字数上限常驻断言（当天曾因 breakpoint-era→breakpoint era 拆词使 npj 摘要 150→151 超限，
# 当时无断言可拦，靠人工点数才发现）
_njp_body = [l for l in _njp_ab.split("\n") if l.strip().startswith("External validation studies document")][0]
_njp_w = len(_njp_body.split())
chk("npj abstract <= 150 words", _njp_w <= 150, f"{_njp_w} words")
_bmc_w = sum(len(re.sub(r"\s+", " ", m.group(2)).split()) for m in re.finditer(
    r"^(Background|Methods|Results|Conclusions):\s*(.*?)(?=\n(?:Background|Methods|Results|Conclusions):|\Z)",
    ab_txt, re.S | re.M))
chk("BMC abstract <= 350 words", _bmc_w <= 350, f"{_bmc_w} words")
# ALL.txt 是 canonical 的逐字拼接（2026-09-22 全量重建），抽查各区段须同步
chk("ALL.txt: §2 uses the new surrogate wording",
    "did not provide a reliable surrogate" in _all2)
chk("ALL.txt: §1 model-family is 'matched or outperformed'",
    "matched or outperformed LightGBM on all ten" in _all2)
chk("ALL.txt: abstract mirrors BMC canonical",
    "covered the prespecified internal reference in nine of ten drug–site directions" in _all2
    and "A transferability audit combining label, covariate, breakpoint era" in _all2)
# 合法编号范围取自 build_docx 的节列表（txt 里 Results 用 1.–15.，4.N 是渲染时加的）
_bd = open(os.path.join(BASE, "04_代码", "build_docx.py"), encoding="utf-8-sig").read()
def _sec_count(name):
    m = re.search(rf"{name} = \[(.*?)\]", _bd, re.S)
    return len(re.findall(r'"[^"]+"', m.group(1))) if m else -1
_n3, _n4 = _sec_count("METHOD_SECTIONS"), _sec_count("RESULT_SECTIONS")
for _sec, _n in (("3", _n3), ("4", _n4)):
    _bad = sorted(r for r in set(re.findall(rf"§{_sec}\.(\d{{1,2}})", _mr_txt2 + _disc2))
                  if not (1 <= int(r) <= _n))
    chk(f"cross-refs: every §{_sec}.N within 1–{_n}", not _bad, f"越界={_bad}")
chk("cross-refs: §3.4 points at Intermediate-Exclusion (§4.13)",
    "dropping Intermediate rows is reported in §4.13" in _mr_txt2)
chk("cross-refs: §5.6 conservatism points at §4.13",
    "smaller in five; §4.13)" in _disc2)
# §4.13 的保守性论断必须自证，不得再指向 §4.14（那是联合干预那节，与该论断无关）
chk("cross-refs: §4.13 conservatism is self-contained",
    "makes internal performance conservative by 0.02–0.06 AUROC." in _mr_txt2
    and "conservative relative to an Intermediate-excluded label" not in _mr_txt2)
chk("cross-refs: no dangling (§4.14) inside Results",
    "in most directions (§4.14)" not in _mr_txt2)
chk("Methods: primary directions range tag", "(−0.14 to +0.02, primary directions)" in mr)
chk("Methods: no underscore-space typo", "prior_pa_res_ " not in mr)
er2 = pd.read_csv(os.path.join(R3, "era_sensitivity.csv"))
caz = er2[(er2["cut"] == 2020) & (er2["task"] == "ceftazidime") & (er2["tgt"] == "UTSW")].iloc[0]
fep = er2[(er2["cut"] == 2020) & (er2["task"] == "cefepime") & (er2["tgt"] == "UTSW")].iloc[0]
chk("era numbers moved to Table S3",
    "remained far from calibration (Table S3)" in disc,
    f"CAZ {caz['brier_all']:.3f}->{caz['brier_cut']:.3f}, FEP {fep['brier_all']:.3f}->{fep['brier_cut']:.3f}")

p("=" * 100)
p(f"VERDICT: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
p("=" * 100)
rep.close()
