# -*- coding: utf-8 -*-
"""阶段③：全量对比评测（64题 × 9组配置）
单变量设计，对比链条：
  检索方式：G01 向量 vs G02 关键词 vs G03 混合（k5/无精排）
  TopK：   G04 k3 vs G01 k5 vs G05 k10（向量/无精排）
  Rerank： G03 混合无精排 vs G06 混合+精排（无兜底）
  兜底阈值：G07 0.3 vs G09 0.4(最佳) vs G08 0.5（混合+精排）
  调整前后总对比：G01 → G09
"""
import sys
from pathlib import Path

sys.path[:0] = [str(Path(__file__).resolve().parent.parent), str(Path(__file__).resolve().parent)]

from eval_lib import build_components, load_samples, run_group

CONFIGS = [
    {"name": "G01_基线_向量_k5_无精排", "mode": "向量检索", "top_k": 5, "rerank": False,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G02_检索_关键词_k5", "mode": "关键词", "top_k": 5, "rerank": False,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G03_检索_混合_k5", "mode": "混合检索", "top_k": 5, "rerank": False,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G04_TopK3_向量", "mode": "向量检索", "top_k": 3, "rerank": False,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G05_TopK10_向量", "mode": "向量检索", "top_k": 10, "rerank": False,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G06_混合_Rerank_无兜底", "mode": "混合检索", "top_k": 5, "rerank": True,
     "rerank_k": 10, "fallback_on": False, "fallback_th": 0.4, "threshold": 0.0},
    {"name": "G07_兜底阈值0.3", "mode": "混合检索", "top_k": 5, "rerank": True,
     "rerank_k": 10, "fallback_on": True, "fallback_th": 0.3, "threshold": 0.0},
    {"name": "G08_兜底阈值0.5", "mode": "混合检索", "top_k": 5, "rerank": True,
     "rerank_k": 10, "fallback_on": True, "fallback_th": 0.5, "threshold": 0.0},
    {"name": "G09_优化后最佳_混合k5精排兜底0.4", "mode": "混合检索", "top_k": 5, "rerank": True,
     "rerank_k": 10, "fallback_on": True, "fallback_th": 0.4, "threshold": 0.0},
]


def main():
    args = sys.argv[1:]
    workers = 6
    if "--workers" in args:
        workers = int(args[args.index("--workers") + 1])
    # 位置参数 = 只跑指定配置名；--workers 及其数值不作为配置名
    only = [a for a in args if not a.startswith("--")]
    if "--workers" in args:
        wi = args.index("--workers")
        only = [a for i, a in enumerate(args) if not a.startswith("--") and i != wi + 1]
    configs = [c for c in CONFIGS if not only or c["name"] in only]
    samples = load_samples()
    print(f"全量评测：{len(samples)} 题 × {len(configs)} 组，并发 {workers}")
    comp = build_components()
    for gcfg in configs:
        run_group(comp, samples, gcfg, workers=workers)
    print("全部配置组评测完成。")


if __name__ == "__main__":
    main()
