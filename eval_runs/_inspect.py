# -*- coding: utf-8 -*-
import json
import sys
from pathlib import Path

sys.path[:0] = [".", "eval_runs"]
from eval_lib import GOLD_IDS, load_samples

recs = {r["sid"]: r for r in json.loads(
    Path("eval_runs/results/calibration_best.json").read_text(encoding="utf-8"))["records"]}
samples = {s["sid"]: s for s in load_samples()}

for sid in GOLD_IDS:
    r, s = recs[sid], samples[sid]
    h, j = r.get("hard") or {}, r.get("judge") or {}
    print(f"--- {sid} [{s['category']}] 期望:{s['expected_intent']}")
    print(f"    Q: {s['question']}")
    print(f"    意图{'OK' if h.get('intent_ok') else 'MISMATCH'} 实际:{h.get('intent_actual')} | "
          f"fallback:{h.get('fallback_triggered')} | reject_ok:{h.get('reject_ok')} | tool_ok:{h.get('tool_ok')} | "
          f"fc:{h.get('fact_coverage')} | 裁判 相关/忠实/检索: {j.get('relevance')}/{j.get('faithfulness')}/{j.get('retrieval_relevance')}")
    print(f"    A: {(h.get('answer') or '')[:110]}")
