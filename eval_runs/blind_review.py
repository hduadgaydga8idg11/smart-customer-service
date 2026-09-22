# -*- coding: utf-8 -*-
"""阶段⑥：12 题盲评抽测
从基线 G01 与最佳 G09 同题对比，隐藏配置名（A/B 乱序），导出人工盲评表。
分层：RAG正确4 + RAG兜底/边界3 + 工具3 + 情绪/对抗2

输出两张表（均为 utf-8-sig，Excel 直接打开）：
  盲评打分表_逐版本.csv  —— 每题 A/B 各占一行，回答竖着排，方便逐句阅读打分
  盲评总评_逐题.csv      —— 每题一行，只填整体偏好
映射（评分期间保密，揭盲时使用）：盲评映射表.json
"""
import csv
import json
import random
import sys
from pathlib import Path

sys.path[:0] = [".", "eval_runs"]
from eval_lib import RESULT_DIR, load_samples

samples = {s["sid"]: s for s in load_samples()}
random.seed(42)


def load(name):
    p = RESULT_DIR / f"{name}.json"
    return {r["sid"]: r for r in json.loads(p.read_text(encoding="utf-8"))["records"]}


def main():
    base, best = load("G01_基线_向量_k5_无精排"), load("G09_优化后最佳_混合k5精排兜底0.4")
    rag_good = ["03", "13", "14", "56"]   # 常规RAG
    rag_edge = ["26", "39", "52"]         # 边界/兜底/口语
    tools = ["27", "28", "47"]            # 工具（含隐私拦截）
    emo = ["45", "61"]                    # 情绪/对抗
    pick = rag_good + rag_edge + tools + emo

    detail_rows, pref_rows, key = [], [], {}
    for sid in pick:
        s = samples[sid]
        rb, ro = base[sid], best[sid]
        a_is_best = random.choice([True, False])
        a_rec, b_rec = (ro, rb) if a_is_best else (rb, ro)
        key[sid] = {"A": "最佳G09" if a_is_best else "基线G01",
                    "B": "基线G01" if a_is_best else "最佳G09"}
        for ver, rec in (("A", a_rec), ("B", b_rec)):
            detail_rows.append({
                "题号": sid,
                "题型": s["category"],
                "问题": s["question"],
                "标准答案要点": s["ground_truth"],
                "版本": ver,
                "回答": (rec.get("hard") or {}).get("answer", ""),
                "相关性(1-5)": "",
                "事实正确性(对/半对/错/合理拒答)": "",
                "有无编造(无/有)": "",
                "帮助性(1-5)": "",
                "备注": "",
            })
        pref_rows.append({
            "题号": sid,
            "问题": s["question"],
            "整体更好(A/B/持平)": "",
            "一句话理由": "",
        })

    out_dir = RESULT_DIR.parent
    p1 = out_dir / "盲评打分表_逐版本.csv"
    with open(p1, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(detail_rows[0].keys()))
        w.writeheader()
        w.writerows(detail_rows)
    p2 = out_dir / "盲评总评_逐题.csv"
    with open(p2, "w", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(pref_rows[0].keys()))
        w.writeheader()
        w.writerows(pref_rows)
    (out_dir / "盲评映射表.json").write_text(
        json.dumps(key, ensure_ascii=False, indent=1), encoding="utf-8")

    # 旧版单表清理
    old = out_dir / "盲评抽测表.csv"
    if old.exists():
        old.unlink()

    print("已生成：")
    print(" 1)", p1, "（24 行：12题×A/B，每行一个回答）")
    print(" 2)", p2, "（12 行：每题填整体偏好）")
    print("填完后运行：python eval_runs/盲评统计.py  自动揭盲并统计")


if __name__ == "__main__":
    main()
