# -*- coding: utf-8 -*-
"""参数真实性批量核查：前端配置 → 后端实际执行 是否一致"""
import json
import sys
from collections import Counter
from pathlib import Path

sys.path[:0] = [".", "eval_runs"]
from eval_lib import RESULT_DIR, load_samples

samples = {s["sid"]: s for s in load_samples()}

import importlib.util
spec = importlib.util.spec_from_file_location("fe", "eval_runs/full_eval.py")
fe = importlib.util.module_from_spec(spec)
spec.loader.exec_module(fe)

fail = []
for gcfg in fe.CONFIGS:
    p = RESULT_DIR / f"{gcfg['name']}.json"
    if not p.exists():
        print(f"[skip] {gcfg['name']} 未完成")
        continue
    recs = json.loads(p.read_text(encoding="utf-8"))["records"]
    rag = [r for r in recs if (r.get("hard") or {}).get("intent_actual") == "知识库咨询"]
    if len(rag) < 20:
        print(f"[warn] {gcfg['name']} 知识题仅 {len(rag)}")
    # 1) 检索方式：知识题 route_note 首段必须 == 配置模式
    modes = Counter((r["hard"].get("route_note", "").split(" → ")[0]) for r in rag)
    bad_mode = sum(v for k, v in modes.items() if k != gcfg["mode"] and "放行" not in k)
    # 放行标记含模式前缀的也算对（"混合检索 → 向量强命中放行"首段仍是混合检索，已被split取首段）
    bad_mode = sum(v for k, v in modes.items() if k != gcfg["mode"])
    # 2) TopK：知识题 doc_count 必须 == top_k（BM25 稀疏匹配允许不足 k）
    dc = Counter(r["hard"].get("doc_count") for r in rag)
    if gcfg["mode"] == "关键词":
        bad_k = sum(v for k, v in dc.items() if k is not None and k > gcfg["top_k"])
    else:
        bad_k = sum(v for k, v in dc.items() if k != gcfg["top_k"])
    # 3) Rerank 开关：开启时 trace 应有标记——rec 未存 trace，用兜底/放行 note 间接 + 单独探针
    # 4) 兜底开关与阈值：域外题触发率
    ood = [r for r in recs if samples[r["sid"]]["source_doc"] == "未命中兜底"]
    ood_fb = sum(1 for r in ood if (r.get("hard") or {}).get("fallback_triggered"))
    # 正常知识题误兜底
    norm = [r for r in rag if samples[r["sid"]]["source_doc"].endswith(".md")]
    norm_fb = sum(1 for r in norm if (r.get("hard") or {}).get("fallback_triggered"))
    tag = "OK" if bad_mode == 0 and bad_k == 0 else "❌FAIL"
    if bad_mode or bad_k:
        fail.append(gcfg["name"])
    print(f"[{tag}] {gcfg['name']}")
    print(f"    知识题{len(rag)} 模式错{bad_mode} TopK错{bad_k} doc分布{dict(dc)} | "
          f"域外兜底 {ood_fb}/{len(ood)} | 正常题误兜底 {norm_fb}/{len(norm)}")

print("\n参数真实性结论:", "全部一致 ✅" if not fail else f"失败组: {fail}")
