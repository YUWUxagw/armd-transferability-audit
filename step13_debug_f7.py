# -*- coding: utf-8 -*-
"""调试 F7: 患者 54887 合并症数据流逐段比对"""
import pandas as pd

M = r"E:\ARMD\ARMD-MGB"
CLEAN = r"E:\ARMD\data\clean"
pid = 54887

base = pd.read_csv(rf"{CLEAN}\MGB\site_base_v3.csv", low_memory=False, encoding="utf-8-sig")
base["time"] = pd.to_datetime(base["time"], errors="coerce")
rows = base[base["anon_id"] == pid]
print(f"基座中该患者培养数: {len(rows)} | 培养468623在基座: {int((rows['order_proc_id_coded']==468623).sum())}")
cx = rows[rows["order_proc_id_coded"] == 468623]
print(f"468623 的 time: {cx['time'].iloc[0]} | comorb_count: {int(cx['comorb_count'].iloc[0])}")

# 1) 原始合并症行
cm = pd.read_csv(rf"{M}\comorbidity_deid_tj.csv", low_memory=False, encoding="utf-8-sig")
cm = cm[cm["anon_id"] == pid]
cm["att"] = pd.to_datetime(cm["order_time_jittered_utc_shifted"], errors="coerce")
print(f"\n原始合并症行: {len(cm)} | att 解析失败: {int(cm['att'].isna().sum())}")
print(f"att 样例: {cm['att'].head(3).tolist()}")
print(f"att_oid 唯一数: {cm['order_proc_id_coded'].nunique()}")

# 2) 该患者全量培养 (cohort)
mc = pd.read_csv(rf"{M}\microbiology_cohort_deid_tj_updated.csv",
                 usecols=["anon_id", "order_proc_id_coded", "order_time_jittered_utc_shifted"],
                 low_memory=False, encoding="utf-8-sig")
mc = mc[mc["anon_id"] == pid].drop_duplicates("order_proc_id_coded")
mc["time"] = pd.to_datetime(mc["order_time_jittered_utc_shifted"], errors="coerce")
print(f"\n该患者全量培养: {len(mc)}")

# 合并症挂接培养在患者培养集中吗?
att_ids = set(cm["order_proc_id_coded"])
cult_ids = set(mc["order_proc_id_coded"])
print(f"合并症 att_oid 在患者培养集: {len(att_ids & cult_ids)}/{len(att_ids)}")
print(f"合并症 att_oid 不在患者培养集样例: {list(att_ids - cult_ids)[:5]}")

# 3) 重建该患者 pairs (att<idx)
mc2 = mc.sort_values("time")
pairs = []
ts = mc2["time"].values; ids = mc2["order_proc_id_coded"].values
for i in range(len(mc2)):
    for j in range(i + 1, len(mc2)):
        pairs.append((ids[i], ids[j], int((ts[j] - ts[i]) / pd.Timedelta(days=1))))
pairs = pd.DataFrame(pairs, columns=["att_oid", "idx_cx", "gap"])
print(f"\n该患者培养对: {len(pairs)}")

# 4) hist_events 等价合并 (min_gap=1)
m = cm.merge(pairs, on="att_oid")
m = m[(m["gap"] >= 1)]
print(f"合并后行数: {len(m)} | 468623 为索引的行: {int((m['idx_cx']==468623).sum())}")
cnt = m[m["idx_cx"] == 468623].groupby("idx_cx").size()
print(f"清洗逻辑下 468623 的 comorb_count: {int(cnt.sum()) if len(cnt) else 0}")

# 5) 对照 spot-check 逻辑 (直接时间比较, 任意更早培养)
m2 = cm.merge(rows[["order_proc_id_coded", "time"]].rename(columns={"order_proc_id_coded": "idx_cx", "time": "idx_t"}), on="anon_id")
m2 = m2[(m2["idx_t"] > m2["att"]) & (m2["order_proc_id_coded"] != m2["idx_cx"])]
cnt2 = m2[m2["idx_cx"] == 468623]
print(f"spot-check 逻辑下 468623 的行数: {len(cnt2)}")

# 6) 找分歧: 清洗逻辑漏掉的行
m_ids = set(m[m["idx_cx"] == 468623]["att_oid"])
m2_ids = set(cnt2["order_proc_id_coded"])
print(f"\n清洗逻辑覆盖的挂接培养数: {len(m_ids)} | spot-check 覆盖: {len(m2_ids)}")
print(f"spot-check 有而清洗无的挂接培养: {list(m2_ids - m_ids)[:5]}")
if m2_ids - m_ids:
    miss = list(m2_ids - m_ids)[0]
    print(f"样例挂接培养 {miss}: 在患者培养集={miss in cult_ids} | 在 pairs 的 att_oid 集={miss in set(pairs['att_oid'])}")
