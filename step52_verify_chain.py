# -*- coding: utf-8 -*-
"""
Step 52 — 全链回归验证 (应检尽检: 任何从 raw 重建队列的分析必须通过此链)
断言链:
  A1 特征集一致: build() 输出特征列集 == 规范 task 文件特征列集 (每任务)
  A2 primary 复现: build() primary (无修复) 的 n 与 AUROC 精确复现 transfer_matrix
     规范值 (10 方向, 独立重算; 容差 0.002)
  A3 台账自洽: attribution_ledger.fixed_auc == joint_fix_ci.auroc (10 方向)
  A4 组合自洽: pairwise.R1R2R3_gap == joint_fix_ci.gap (10 方向)
  A5 primary 自洽: pairwise.primary_gap == transfer_matrix 推算 gap (10 方向)
  A6 CI 合法性: ci_lo < auroc < ci_hi (10 方向)
  A7 单修/组合行内约束: joint 组合在所有 pairwise 组合列中不缺失
运行: python step52_verify_chain.py  (约 3-5 分钟, 输出 report_verify_chain.txt)
"""
import os, sys, ast
import numpy as np
import pandas as pd

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "data", "clean")
OUT = os.path.join(BASE, "05_源数据", "phase3")
rep = open(os.path.join(OUT, "report_verify_chain.txt"), "w", encoding="utf-8")
def p(*a):
    line = " ".join(str(x) for x in a)
    print(line, flush=True); rep.write(line + "\n"); rep.flush()

import warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.join(BASE, "04_代码"))
from step14_model_phase2 import train_lgb, fit_cat_maps, feature_cols, prep, DROP_COLS, EXCLUDE_FEATURES
from sklearn.metrics import roc_auc_score

def _load_internal():
    """内参运行时从权威结果文件读取，禁止字面量（2026-09-18 修复）。

    原字面量 {0.795,0.765,0.764,0.804,0.790} 是 early-stopping 泄漏修复前的
    膨胀值，与 transfer_matrix.csv 的 internal_auroc 不符；按归档脚本重跑会
    系统性偏离已发表数字（例如 MEM→S 的 −0.112 会被算成 −0.136）。
    """
    _tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
    _out = {}
    for _t in ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]:
        _v = _tm[_tm["task"] == _t]["internal_auroc"].unique()
        assert len(_v) == 1, f"internal_auroc not unique for {_t}"
        _out[_t] = float(_v[0])
    return _out


INTERNAL = _load_internal()TASKS = ["meropenem","ciprofloxacin","levofloxacin","ceftazidime","cefepime"]
SITES = ["Stanford","UTSW"]

# --- 从 step51 源文件 AST 提取 build/encode 及其模块级常量 (零漂移) ---
src = open(os.path.join(BASE, "04_代码", "step51_pairwise_fixes.py"), encoding="utf-8").read()
tree = ast.parse(src)
ns = {"os": os, "np": np, "pd": pd, "sys": sys}
CONST_NAMES = ("BASE", "CLEAN", "OUT", "TASKS", "COHORT", "PHENO", "POSCOL", "NEGCOL", "PRELIM", "INTERNAL")
for node in tree.body:
    if isinstance(node, ast.FunctionDef) and node.name in ("build", "encode"):
        exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
    elif isinstance(node, ast.Assign):
        for tgt in node.targets:
            if isinstance(tgt, ast.Name) and tgt.id in CONST_NAMES:
                exec(compile(ast.Module(body=[node], type_ignores=[]), "<step51>", "exec"), ns)
build, encode = ns["build"], ns["encode"]

tm = pd.read_csv(os.path.join(OUT, "transfer_matrix.csv"))
ledger = pd.read_csv(os.path.join(OUT, "attribution_ledger.csv"))
jc = pd.read_csv(os.path.join(OUT, "joint_fix_ci.csv"))
pw = pd.read_csv(os.path.join(OUT, "pairwise_fixes.csv"))

fails = []
def check(name, cond, detail=""):
    p(f"  [{'PASS' if cond else 'FAIL'}] {name} {detail}")
    if not cond: fails.append(name)

p("=" * 90)
p("Step 52 — full-chain regression verification (rebuilt cohort vs canonical)")
p("=" * 90)

# ---------- A1 特征集一致 ----------
p("\n[A1] Feature-set consistency: build() output vs canonical task files")
for t in TASKS:
    tf = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    canon = set(feature_cols(prep(tf)))
    tr = encode(build("MGB", t, False, False, None),
                fit_cat_maps(tf), tf["adi"].median())
    rebuilt = set(feature_cols(tr))
    check(f"A1 {t}", rebuilt == canon,
          f"| rebuilt={len(rebuilt)} canon={len(canon)} diff={sorted(rebuilt ^ canon)}")

# ---------- A2 primary 复现 (独立重算) ----------
p("\n[A2] Primary reproduction: rebuilt primary vs canonical transfer_matrix (recomputed)")
for t in TASKS:
    mgb_raw = pd.read_csv(os.path.join(CLEAN, "MGB", f"task_{t}.csv"), low_memory=False, encoding="utf-8-sig")
    maps = fit_cat_maps(mgb_raw); adi = mgb_raw["adi"].median()
    tr = encode(build("MGB", t, False, False, None), maps, adi)
    feats = feature_cols(tr)
    m = train_lgb(tr[feats], tr["label"], tr[feats], tr["label"])
    for s in SITES:
        te = encode(build(s, t, False, False, None), maps, adi)
        X = te[[f for f in feats if f in te.columns]]
        auc = roc_auc_score(te["label"], m.predict_proba(X)[:, 1])
        rr = tm[(tm["task"] == t) & (tm["src"] == "MGB") & (tm["tgt"] == s)].iloc[0]
        n_ok = len(te) == rr["n_test"]
        auc_ok = abs(auc - rr["auroc"]) <= 0.002
        check(f"A2 {t}->{s[:1]}", n_ok and auc_ok,
              f"| rebuilt n={len(te)} canon n={int(rr['n_test'])}; "
              f"rebuilt AUC={auc:.4f} canon={rr['auroc']:.4f}")

# ---------- A3 台账 self-consistency ----------
p("\n[A3] Ledger self-consistency: attribution_ledger.fixed_auc == joint_fix_ci.auroc")
for _, r in ledger.iterrows():
    j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
    check(f"A3 {r['task']}->{r['tgt'][:1]}", abs(r["fixed_auc"] - j["auroc"]) <= 0.001,
          f"| ledger {r['fixed_auc']:.3f} joint_ci {j['auroc']:.3f}")

# ---------- A4 pairwise joint == joint CI ----------
p("\n[A4] Pairwise joint == joint_fix_ci (same cohort, same encoding)")
for _, r in pw.iterrows():
    j = jc[(jc["task"] == r["task"]) & (jc["tgt"] == r["tgt"])].iloc[0]
    gap = round(j["auroc"] - INTERNAL[r["task"]], 3)
    check(f"A4 {r['task']}->{r['tgt'][:1]}", gap == r["R1R2R3_gap"],
          f"| joint_ci gap {gap:+.3f} pairwise {r['R1R2R3_gap']:+.3f}")

# ---------- A5 pairwise primary == canonical gap ----------
p("\n[A5] Pairwise primary gap == canonical transfer_matrix gap")
for _, r in pw.iterrows():
    rr = tm[(tm["task"] == r["task"]) & (tm["src"] == "MGB") & (tm["tgt"] == r["tgt"])].iloc[0]
    gap = round(rr["auroc"] - INTERNAL[r["task"]], 3)
    check(f"A5 {r['task']}->{r['tgt'][:1]}", gap == r["primary_gap"],
          f"| canon gap {gap:+.3f} pairwise {r['primary_gap']:+.3f}")

# ---------- A6 CI legality ----------
p("\n[A6] CI legality: ci_lo < auroc < ci_hi")
for _, r in jc.iterrows():
    ok = r["ci_lo"] < r["auroc"] < r["ci_hi"]
    check(f"A6 {r['task']}->{r['tgt'][:1]}", ok,
          f"| {r['auroc']:.3f} [{r['ci_lo']:.3f}-{r['ci_hi']:.3f}]")

# ---------- A7 joint present in all pairwise rows ----------
p("\n[A7] Joint (R1R2R3) present in all pairwise rows")
check("A7 R1R2R3_gap non-null everywhere",
      pw["R1R2R3_gap"].notna().all() and len(pw) == 10)

p("\n" + "=" * 90)
p(f"RESULT: {'ALL PASS' if not fails else 'FAILURES: ' + ', '.join(fails)}")
rep.close()
print("Done", flush=True)
