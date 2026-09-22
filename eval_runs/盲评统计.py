# -*- coding: utf-8 -*-
"""盲评统计：读入人工填好的两张表，揭盲并输出结论。
用法：python eval_runs/盲评统计.py
只统计已填写的行/列，未填部分自动跳过并提示。
"""
import csv
import json
import statistics as st
import sys
from pathlib import Path

DIR = Path(__file__).parent


def read_csv(name):
    """兼容 UTF-8(BOM) 与 Excel 另存的 ANSI/GBK 编码"""
    path = DIR / name
    for enc in ("utf-8-sig", "gbk"):
        try:
            with open(path, encoding=enc, newline="") as f:
                return list(csv.DictReader(f))
        except UnicodeDecodeError:
            continue
    raise UnicodeDecodeError(f"{path.name}: 无法按 utf-8/gbk 解码，请确认文件编码")


def num(v):
    try:
        return float(str(v).strip())
    except (TypeError, ValueError):
        return None


def main():
    key = json.loads((DIR / "盲评映射表.json").read_text(encoding="utf-8"))
    detail = read_csv("盲评打分表_逐版本.csv")
    pref = read_csv("盲评总评_逐题.csv")

    # ---- 逐版本打分：按真实配置归组 ----
    groups = {"基线G01": {"rel": [], "help": [], "fab": 0, "n": 0},
              "最佳G09": {"rel": [], "help": [], "fab": 0, "n": 0}}
    judged_rows = 0
    for r in detail:
        cfg = key[r["题号"]][r["版本"]]
        g = groups[cfg]
        g["n"] += 1
        rel, helpv = num(r.get("相关性(1-5)")), num(r.get("帮助性(1-5)"))
        if rel is not None:
            g["rel"].append(rel)
        if helpv is not None:
            g["help"].append(helpv)
        if str(r.get("有无编造(无/有)", "")).strip() == "有":
            g["fab"] += 1
        if rel is not None or helpv is not None:
            judged_rows += 1

    print("=" * 56)
    print(f"逐版本打分（已评分行 {judged_rows}/24）")
    print("-" * 56)
    for cfg, g in groups.items():
        rel = f"{st.mean(g['rel']):.2f}/5" if g["rel"] else "未填"
        hp = f"{st.mean(g['help']):.2f}/5" if g["help"] else "未填"
        print(f"{cfg:8s}  相关性 {rel:>7s}   帮助性 {hp:>7s}   标注编造 {g['fab']} 次")

    # ---- 逐题整体偏好：揭盲 ----
    win = {"最佳G09": 0, "基线G01": 0, "持平": 0}
    got = 0
    rows = []
    for r in pref:
        pick = str(r.get("整体更好(A/B/持平)", "")).strip().upper()
        if pick not in ("A", "B"):
            if "平" in pick:
                win["持平"] += 1
                got += 1
                rows.append((r["题号"], "持平"))
            continue
        cfg = key[r["题号"]][pick]
        win[cfg] += 1
        got += 1
        rows.append((r["题号"], cfg))
    print("=" * 56)
    print(f"整体偏好（已填 {got}/12）：最佳G09 胜 {win['最佳G09']} ｜ "
          f"基线G01 胜 {win['基线G01']} ｜ 持平 {win['持平']}")
    if got:
        print("逐题揭盲：", "，".join(f"{s}→{c.replace('G0', ' G0')}" for s, c in rows))

    # ---- 与 L1 裁判（reasoner）一致性 ----
    res = {}
    for tag, fn in (("基线G01", "G01_基线_向量_k5_无精排"),
                    ("最佳G09", "G09_优化后最佳_混合k5精排兜底0.4")):
        recs = json.loads((DIR / "results" / f"{fn}.json").read_text(encoding="utf-8"))["records"]
        res[tag] = {x["sid"]: (x.get("judge") or {}).get("relevance") for x in recs}
    agree, tot = 0, 0
    for r in pref:
        pick = str(r.get("整体更好(A/B/持平)", "")).strip().upper()
        if pick not in ("A", "B"):
            continue
        other = "B" if pick == "A" else "A"
        human_cfg = key[r["题号"]][pick]
        other_cfg = key[r["题号"]][other]
        j1, j2 = res[human_cfg].get(r["题号"]), res[other_cfg].get(r["题号"])
        if j1 is not None and j2 is not None:
            tot += 1
            if j1 >= j2:
                agree += 1
    if tot:
        print("-" * 56)
        print(f"人工偏好与 L1 裁判同向：{agree}/{tot} = {agree/tot:.0%}")
    print("=" * 56)
    if got < 12:
        print("提示：还有题目未完成评分，当前为部分统计。")


if __name__ == "__main__":
    main()
