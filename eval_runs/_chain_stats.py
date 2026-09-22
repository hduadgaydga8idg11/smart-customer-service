# -*- coding: utf-8 -*-
"""全链路结果分析：路由准确率 / 归因分布 / 节点耗时 / token"""
import json
import statistics as st
from collections import Counter
from pathlib import Path

recs = json.loads(Path("eval_runs/results/chain_best.json").read_text(encoding="utf-8"))["records"]
recs = [r for r in recs if not r.get("error")]
n = len(recs)

route_ok = sum(1 for r in recs if r["route_status"] == "ok")
attr = Counter(r["attribution"] for r in recs)
times = [r["total_time"] for r in recs]
sems = [r["semantic_similarity"] for r in recs if r["semantic_similarity"] is not None]

# 各意图平均耗时
by_intent_time = {}
for r in recs:
    by_intent_time.setdefault(r["predicted_intent"], []).append(r["total_time"])

# token 汇总
total_tokens = 0
for r in recs:
    for node, u in (r.get("llm_usage") or {}).items():
        total_tokens += (u.get("total_tokens") or 0)

print(f"有效样本 {n}")
print(f"意图路由准确率: {route_ok}/{n} = {route_ok/n:.1%}")
print(f"答案-标准义语义相似度均值: {st.mean(sems):.3f}")
print(f"端到端耗时: 中位 {st.median(times):.1f}s | 均值 {st.mean(times):.1f}s | P90 {sorted(times)[int(0.9*n)]:.1f}s")
print(f"总 token 消耗: {total_tokens} | 单题均值 {total_tokens//n}")
print("\n各路由平均耗时:")
for k, v in sorted(by_intent_time.items()):
    print(f"  {k}: {st.mean(v):.1f}s（{len(v)}题）")
print("\n错误归因分布:")
for k, v in attr.most_common():
    print(f"  {v:3d}  {k}")

# 节点级耗时（从 node_runs 提取）
node_times = {}
for r in recs:
    for nr in r.get("node_runs", []):
        node_times.setdefault(nr["node"], []).append(nr.get("latency", 0) or 0)
print("\n节点平均耗时:")
for k, v in sorted(node_times.items(), key=lambda x: -st.mean(x[1])):
    if st.mean(v) > 0.05:
        print(f"  {k}: {st.mean(v):.2f}s")
