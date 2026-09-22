# -*- coding: utf-8 -*-
"""阶段⑤：汇总 9 组结果 → 对比实验表（L0 硬指标 + L1 裁判分）"""
import json
import sys
from pathlib import Path

sys.path[:0] = [".", "eval_runs"]
from eval_lib import RESULT_DIR, load_samples
from core.chain_eval import normalize_intent

LABELS = {
    "G01_基线_向量_k5_无精排": "基线：向量/TopK5/无精排",
    "G02_检索_关键词_k5": "关键词检索",
    "G03_检索_混合_k5": "混合检索",
    "G04_TopK3_向量": "TopK=3",
    "G05_TopK10_向量": "TopK=10",
    "G06_混合_Rerank_无兜底": "混合+Rerank",
    "G07_兜底阈值0.3": "Rerank+兜底0.3",
    "G08_兜底阈值0.5": "Rerank+兜底0.5",
    "G09_优化后最佳_混合k5精排兜底0.4": "优化后：混合+Rerank+兜底0.4",
}
ORDER = list(LABELS.keys())
samples = {s["sid"]: s for s in load_samples()}


def avg(xs):
    xs = [x for x in xs if x is not None]
    return round(sum(xs) / len(xs), 2) if xs else None


def summarize(name):
    path = RESULT_DIR / f"{name}.json"
    if not path.exists():
        return None
    recs = json.loads(path.read_text(encoding="utf-8"))["records"]
    n = len(recs)
    errors = sum(1 for r in recs if r.get("error"))
    intent_ok, fc_all, tool_ok, reject_ok = [], [], [], []
    judge_rel, judge_faith, judge_ret = [], [], []
    intent_total = tool_total = reject_total = 0
    # 兜底专项：域外题是否触发兜底、正常知识题是否误兜底
    ood_triggered = ood_total = norm_wrong_fb = norm_total = 0

    for r in recs:
        s = samples[r["sid"]]
        h, j = r.get("hard"), r.get("judge")
        if not h:
            continue
        intent_ok.append(1 if h.get("intent_ok") else 0)
        intent_total += 1
        if h.get("fact_coverage") is not None and s["source_doc"].endswith(".md"):
            fc_all.append(h["fact_coverage"])
        if h.get("tool_ok") is not None:
            tool_total += 1
            tool_ok.append(1 if h["tool_ok"] else 0)
        if h.get("reject_ok") is not None:
            reject_total += 1
            reject_ok.append(1 if h["reject_ok"] else 0)
        if j:
            judge_rel.append(j.get("relevance"))
            judge_faith.append(j.get("faithfulness"))
            judge_ret.append(j.get("retrieval_relevance"))
        # 兜底专项统计
        is_ood = s["source_doc"] == "未命中兜底"
        if is_ood:
            ood_total += 1
            # 安全处置：置信度兜底，或路由到闲聊做婉拒（均未编造）
            if h.get("fallback_triggered") or h.get("intent_actual") == "日常闲聊" or h.get("reject_ok"):
                ood_triggered += 1
        elif s["source_doc"].endswith(".md"):
            norm_total += 1
            if h.get("fallback_triggered"):
                norm_wrong_fb += 1

    return {
        "n": n, "errors": errors,
        "意图准确率%": round(100 * sum(intent_ok) / intent_total, 1) if intent_total else None,
        "事实覆盖率": avg(fc_all),
        "工具正确率%": round(100 * sum(tool_ok) / tool_total, 1) if tool_total else None,
        "域外安全处置率%": round(100 * ood_triggered / ood_total, 1) if ood_total else None,
        "正常题误兜底%": round(100 * norm_wrong_fb / norm_total, 1) if norm_total else None,
        "答案相关性": avg(judge_rel),
        "忠实度": avg(judge_faith),
        "检索裁判分": avg([x for x in judge_ret if x is not None and x >= 0]),
    }


def main():
    rows = []
    for name in ORDER:
        m = summarize(name)
        if m:
            m["配置"] = LABELS[name]
            rows.append(m)
    if not rows:
        print("暂无组结果")
        return
    cols = ["配置", "n", "意图准确率%", "事实覆盖率", "答案相关性", "忠实度", "检索裁判分",
            "工具正确率%", "域外安全处置率%", "正常题误兜底%", "errors"]
    print("| " + " | ".join(cols) + " |")
    print("|" + "---|" * len(cols))
    for r in rows:
        print("| " + " | ".join(str(r.get(c, "")) for c in cols) + " |")
    out = RESULT_DIR / "summary.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\nsaved:", out)


if __name__ == "__main__":
    main()
