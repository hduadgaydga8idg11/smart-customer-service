# -*- coding: utf-8 -*-
"""阶段④：Agent 全链路评测（最佳配置，64题，真实生产图）
指标：意图路由准确率、节点状态（改写/检索/工具/回复）、逐节点耗时、token、裁判节点分。
"""
import json
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from core.chain_eval import run_chain_eval_single
from eval_lib import RESULT_DIR, build_components, load_samples

BEST_CFG = {"mode": "混合检索", "top_k": 5, "rerank": True, "rerank_k": 10,
            "fallback_on": True, "fallback_th": 0.4, "threshold": 0.0}
OUT = RESULT_DIR / "chain_best.json"


def main():
    workers = int(sys.argv[sys.argv.index("--workers") + 1]) if "--workers" in sys.argv else 6
    samples = load_samples()
    comp = build_components()
    deps = {"graph": comp["graph"], "chat_model": comp["sut"], "embeddings": comp["embeddings"]}

    done = {}
    if OUT.exists():
        done = {r["sid"]: r for r in json.loads(OUT.read_text(encoding="utf-8")).get("records", [])}
    todo = [s for s in samples if s["sid"] not in done]
    recs = list(done.values())
    print(f"全链路评测：共 {len(samples)}，已完成 {len(done)}，待跑 {len(todo)}")

    for i in range(0, len(todo), workers):
        batch = todo[i:i + workers]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {
                ex.submit(run_chain_eval_single, deps, s["question"],
                          s["expected_intent"], s["ground_truth"], BEST_CFG): s
                for s in batch
            }
            for fu in as_completed(futs):
                s = futs[fu]
                try:
                    r = fu.result()
                    r["sid"] = s["sid"]
                    r["category"] = s["category"]
                    # Document 对象不可 JSON 序列化：替换为数量+来源；节点更新里的 docs 一并剔除
                    r["doc_count"] = len(r.get("docs", []) or [])
                    r["doc_sources"] = sorted({d.metadata.get("source", "?") for d in (r.get("docs") or [])})
                    r.pop("docs", None)
                    for nr in r.get("node_runs", []):
                        upd = nr.get("update") or {}
                        upd.pop("docs", None)
                    recs.append(r)
                except Exception as e:
                    recs.append({"sid": s["sid"], "error": str(e)[:200]})
        recs.sort(key=lambda r: r["sid"])
        OUT.write_text(json.dumps({"config": BEST_CFG, "records": recs}, ensure_ascii=False, indent=1),
                       encoding="utf-8")
        print(f"  进度 {len(recs)}/{len(samples)}", flush=True)
    print("全链路评测完成：", OUT)


if __name__ == "__main__":
    main()
