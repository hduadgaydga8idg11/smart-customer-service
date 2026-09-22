# -*- coding: utf-8 -*-
"""阶段②：20 题金标集裁判校准
用最佳配置（混合/k5/Rerank/兜底0.4）跑 20 题 → reasoner 裁判 + L0 硬指标
→ 输出 calibration.md / .csv（含"建议判定"列，用户过目改成"人工判定"）
一致率 >= 16/20 (80%) 才允许进入全量评测。
"""
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
sys.path.insert(0, str(Path(__file__).resolve().parent))

from eval_lib import GOLD_IDS, RESULT_DIR, build_components, load_samples, run_group

BEST = {"name": "calibration_best", "mode": "混合检索", "top_k": 5, "rerank": True,
        "rerank_k": 10, "fallback_on": True, "fallback_th": 0.4, "threshold": 0.0}


def suggest_verdict(rec, sample) -> str:
    hard, judge = rec.get("hard"), rec.get("judge")
    if rec.get("error") or not hard:
        return "差"
    exp = hard["intent_expected"]
    # 工具题：以工具硬判定为准
    if exp == "查订单":
        return "好" if hard["tool_ok"] else "差"
    if exp == "创建工单":
        return "好" if hard["tool_ok"] else "差"
    # 闲聊题（含情绪安抚）：以裁判答案相关性为准
    if exp == "日常闲聊":
        rel = (judge or {}).get("relevance")
        if rel is None:
            return "差"
        return "好" if rel >= 6 else ("中" if rel >= 4 else "差")
    # 域外拒答题：正确触发兜底/婉拒即判好（行为正确，裁判相关性低是预期的）
    if sample["source_doc"] == "未命中兜底":
        if hard.get("fallback_triggered") or hard.get("reject_ok"):
            return "好"
        rel = (judge or {}).get("relevance")
        return "中" if rel is not None and rel >= 5 else "差"
    # RAG 常规题：裁判相关性 + 事实覆盖
    rel = (judge or {}).get("relevance") or 0
    fc = hard.get("fact_coverage") or 0
    if rel >= 7 and fc >= 0.6:
        return "好"
    if rel >= 4 and fc >= 0.35:
        return "中"
    return "差"


def main():
    samples = [s for s in load_samples() if s["sid"] in GOLD_IDS]
    print(f"金标集 {len(samples)} 题，构建运行时（deepseek-chat 被测 / reasoner 裁判）...")
    comp = build_components()
    recs = run_group(comp, samples, BEST, workers=6)
    by_sid = {r["sid"]: r for r in recs}
    smap = {s["sid"]: s for s in samples}

    rows = []
    for sid in GOLD_IDS:
        s, r = smap[sid], by_sid[sid]
        verdict = suggest_verdict(r, s) if not r.get("error") else "差"
        h, j = r.get("hard") or {}, r.get("judge") or {}
        rows.append({
            "sid": sid, "category": s["category"], "expected_intent": s["expected_intent"],
            "question": s["question"], "ground_truth": s["ground_truth"],
            "answer": h.get("answer", ""), "route_note": h.get("route_note", ""),
            "intent_ok": h.get("intent_ok"), "fact_coverage": h.get("fact_coverage"),
            "tool_ok": h.get("tool_ok"), "reject_ok": h.get("reject_ok"),
            "j_relevance": j.get("relevance"), "j_faith": j.get("faithfulness"),
            "j_retrieval": j.get("retrieval_relevance"), "judge_err": j.get("judge_err"),
            "建议判定": verdict, "人工判定": "",
        })

    # CSV（供用户填"人工判定"列：好/中/差）
    csv_path = RESULT_DIR / "calibration.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)

    # Markdown 对照表（便于直接阅读）
    md = ["# 金标集裁判校准表（20 题）\n",
          "请逐题对照「标准答案」和「实际回答」，在最后一列填写你的判定（好/中/差）。",
          "「建议判定」是机评参考；完成后把人工判定率告诉我即可。\n"]
    for r in rows:
        md.append(f"## {r['sid']}｜{r['category']}｜期望意图：{r['expected_intent']}")
        md.append(f"- **问题**：{r['question']}")
        md.append(f"- **标准答案**：{r['ground_truth']}")
        md.append(f"- **实际回答**：{r['answer']}")
        md.append(f"- 路由：{'✅' if r['intent_ok'] else '❌'} {r['route_note']}")
        md.append(f"- 事实覆盖率：{r['fact_coverage']}｜裁判分（相关/忠实/检索）："
                  f"{r['j_relevance']} / {r['j_faith']} / {r['j_retrieval']}")
        if r["tool_ok"] is not None:
            md.append(f"- 工具执行：{'✅' if r['tool_ok'] else '❌'}")
        if r["reject_ok"] is not None:
            md.append(f"- 域外拒答：{'✅' if r['reject_ok'] else '❌'}")
        md.append(f"- **建议判定：{r['建议判定']}**　｜　你的判定：____")
        md.append("")
    md_path = RESULT_DIR.parent / "calibration.md"
    md_path.write_text("\n".join(md), encoding="utf-8")
    print(f"\n已生成：\n{csv_path}\n{md_path}")
    good = sum(1 for r in rows if r["建议判定"] == "好")
    mid = sum(1 for r in rows if r["建议判定"] == "中")
    bad = sum(1 for r in rows if r["建议判定"] == "差")
    intent_acc = sum(1 for r in rows if r["intent_ok"]) / len(rows)
    print(f"建议判定分布：好 {good} / 中 {mid} / 差 {bad}｜意图准确率 {intent_acc:.0%}")


if __name__ == "__main__":
    main()
