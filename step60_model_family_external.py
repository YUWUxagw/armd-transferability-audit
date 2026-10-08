# -*- coding: utf-8 -*-
"""Step 60 —— 模型家族跨站外部比较（重建 model_family_external*.csv 的生产者）。

背景
----
`05_源数据/phase3/model_family_external.csv` 与 `model_family_external_calibration.csv`
（2026-09-21 16:20/16:28）被稿件引用 —— Discussion 的模型家族段、摘要的
"matched or outperformed … in all ten primary directions" —— 但随包脚本里
**没有任何文件产出它们**，属「有产物无生产者」。本脚本补上这个生产者。

口径（与 step16 跨站主流程一致）
--------------------------------
在**源站全队列**上训练、在**目标站全队列**上评估，**不做任何交叉验证**；
两站之间不存在患者重叠，故没有患者跨折的泄漏路径。CSV 里的 `n_test`
即目标站该药的队列规模（MGB 9684 / Stanford 5250 / UTSW 7213，meropenem）。

- LightGBM 列直接取自冻结的 `transfer_matrix.csv`（不重训，零漂移）；
- LR 列由本脚本现算，用 `step14.train_lr`（C=0.5, max_iter=2000, StandardScaler）；
- 两族相对**同一测试队列**，故 `lr_minus_lgb_auroc` 是干净的家族比较；
- 校准斜率/截距用 `step16.calib_metrics` 的同一定义（logit 上的 Platt 缩放）。

安全
----
默认只写 `--out` 指定的目录（缺省为临时目录），**不碰冻结产物**。
比对通过后再用 `--write` 落盘到 05_源数据/phase3/。

用法
----
    python step60_model_family_external.py            # 写临时目录并比对
    python step60_model_family_external.py --write    # 比对通过后落盘
"""
import os
import sys
import tempfile

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

BASE = r"F:\E\Machine Learning\ARMD"
CLEAN = os.path.join(BASE, "clean", "clean")
R2 = os.path.join(BASE, "05_源数据", "phase2")
R3 = os.path.join(BASE, "05_源数据", "phase3")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from step14_model_phase2 import (prep_cross, fit_cat_maps, feature_cols,
                                 train_lr)


def calib_metrics(y, p):
    """逐字抄自 step16_phase3_transfer.py:58-65（同口径）。

    不 import step16 是因为它在模块层 import shap，而本脚本只需这一个函数。
    """
    from sklearn.linear_model import LogisticRegression
    brier = float(np.mean((p - y) ** 2))
    eps = 1e-6
    lp = np.log(np.clip(p, eps, 1 - eps) / (1 - np.clip(p, eps, 1 - eps)))
    lr = LogisticRegression(max_iter=500)
    lr.fit(lp.reshape(-1, 1), y)
    return brier, float(lr.intercept_[0]), float(lr.coef_[0][0])

TASKS = ["meropenem", "ciprofloxacin", "levofloxacin", "ceftazidime", "cefepime"]
SITES = ["MGB", "Stanford", "UTSW"]
ABBR = {"MGB": "MGB", "Stanford": "S", "UTSW": "U"}
DRUG = {"meropenem": "MER", "ciprofloxacin": "CIP", "levofloxacin": "LEV",
        "ceftazidime": "CAZ", "cefepime": "FEP"}


def load(site, task):
    return pd.read_csv(os.path.join(CLEAN, site, "task_%s.csv" % task),
                       low_memory=False, encoding="utf-8-sig")


def main():
    write = "--write" in sys.argv
    outdir = R3 if write else tempfile.mkdtemp(prefix="step60_")
    print("输出目录:", outdir)

    tm = pd.read_csv(os.path.join(R3, "transfer_matrix.csv"))
    cv = pd.read_csv(os.path.join(R2, "cv_internal.csv"))
    int_lr = cv[cv.model == "lr"].groupby("task")["auroc"].mean().to_dict()
    int_lgb = cv[cv.model == "lgb"].groupby("task")["auroc"].mean().to_dict()

    rows, cal_rows = [], []
    for t in TASKS:
        ref = load("MGB", t)
        maps = fit_cat_maps(ref)
        adi_med = float(ref["adi"].median())
        frames = {s: prep_cross(load(s, t), maps, adi_med) for s in SITES}
        feats = feature_cols(frames["MGB"])

        for src in SITES:
            for tgt in SITES:
                if src == tgt:
                    continue
                tr, te = frames[src], frames[tgt]
                m, sc = train_lr(tr[feats], tr["label"])
                p = m.predict_proba(sc.transform(te[feats]))[:, 1]
                y = te["label"]
                auc = float(roc_auc_score(y, p))
                apr = float(average_precision_score(y, p))
                brier, ci, cs = calib_metrics(y.values, p)

                row = tm[(tm.task == t) & (tm.src == src) & (tm.tgt == tgt)]
                if len(row) != 1:
                    print("  !! transfer_matrix 缺行: %s %s->%s" % (t, src, tgt))
                    continue
                lgb_auc = float(row["auroc"].iloc[0])

                # 内部参考只在**源站为 MGB** 时有定义（MGB 是主队列，其患者级 CV
                # 即稿件的 prespecified internal reference）。反向来源方向
                # （Stanford/UTSW 出发）没有对应的 MGB 内部参考，这三列留空 ——
                # 旧表也是留空的，填上 MGB 的值会把参考系搞错。
                if src == "MGB":
                    ref_lr = round(float(int_lr[t]), 4)
                    ref_lgb = round(float(int_lgb[t]), 4)
                    gap = round(auc - ref_lr, 4)
                else:
                    ref_lr = ref_lgb = gap = None

                rows.append(dict(task=t, src=src, tgt=tgt, model="lr",
                                 n_test=int(len(te)), auroc=round(auc, 4),
                                 auprc=round(apr, 4),
                                 internal_ref=ref_lr,
                                 transfer_gap=gap,
                                 lgb_auroc=round(lgb_auc, 4),
                                 lgb_int_ref=ref_lgb,
                                 lr_minus_lgb_auroc=round(auc - lgb_auc, 4)))
                if src == "MGB":
                    if ABBR[src] != "MGB":
                        pass
                    cal_rows.append(dict(
                        d="%s→%s" % (DRUG[t], ABBR[tgt]),
                        lr_auroc=round(auc, 4), lr_int=round(ci, 4),
                        lr_slope=round(cs, 4),
                        lgb_int=round(float(row["calib_int"].iloc[0]), 4),
                        lgb_slope=round(float(row["calib_slope"].iloc[0]), 4)))

    ext = pd.DataFrame(rows)
    cal = pd.DataFrame(cal_rows)
    f1 = os.path.join(outdir, "model_family_external.csv")
    f2 = os.path.join(outdir, "model_family_external_calibration.csv")
    # utf-8-sig：冻结产物历来带 BOM（本项目其余 CSV 同此约定）
    ext.to_csv(f1, index=False, encoding="utf-8-sig")
    cal.to_csv(f2, index=False, encoding="utf-8-sig")

    # ---- 与已发表版本比对 ----
    def compare(new, old_path, keys, cols, title):
        print("\n" + "=" * 84)
        print(title)
        print("=" * 84)
        if not os.path.exists(old_path):
            print("  已发表版本不存在，无法比对")
            return
        old = pd.read_csv(old_path)
        n = new.merge(old, on=keys, suffixes=("_new", "_old"))
        print("  行数: 新 %d / 旧 %d / 可配对 %d" % (len(new), len(old), len(n)))
        worst, nbad = 0.0, 0
        for c in cols:
            if c + "_new" not in n.columns:
                print("    %-22s 新表无此列" % c); continue
            d = (n[c + "_new"].astype(float) - n[c + "_old"].astype(float)).abs()
            mx = float(d.max())
            nbad += int((d > 5e-5).sum())
            worst = max(worst, mx)
            flag = "OK " if mx <= 5e-5 else "!! "
            print("    %s %-22s 最大差 %.6f   超差行数 %d" % (flag, c, mx, int((d > 5e-5).sum())))
        print("  → %s" % ("逐位复现 ✅" if nbad == 0 else "存在差异（%d 处）" % nbad))
        return nbad

    bad = 0
    bad += compare(ext, os.path.join(R3, "model_family_external.csv"),
                   ["task", "src", "tgt"],
                   ["n_test", "auroc", "auprc", "internal_ref", "transfer_gap",
                    "lgb_auroc", "lgb_int_ref", "lr_minus_lgb_auroc"],
                   "model_family_external.csv")
    bad += compare(cal, os.path.join(R3, "model_family_external_calibration.csv"),
                   ["d"], ["lr_auroc", "lr_int", "lr_slope", "lgb_int", "lgb_slope"],
                   "model_family_external_calibration.csv")

    print("\n" + "=" * 84)
    if write and bad == 0:
        print("比对全部通过，已落盘到", R3)
    elif write:
        print("!! 存在差异，**拒绝落盘**（--write 已在比对失败时失效）")
        sys.exit(1)
    else:
        print("比对模式（未落盘）。确认无误后加 --write。")
    print("=" * 84)


if __name__ == "__main__":
    main()
