# -*- coding: utf-8 -*-
import sys
sys.path[:0] = [".", "eval_runs"]
from langchain_chroma import Chroma
from core.model_factory import build_embeddings, load_model_config
from core.retrieval import set_vectorstore
from eval_lib import GOLD_IDS, load_samples

emb = build_embeddings(load_model_config())
vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
set_vectorstore(vs)
samples = {s["sid"]: s for s in load_samples()}

for sid in GOLD_IDS:
    s = samples[sid]
    try:
        hits = vs.similarity_search_with_score(s["question"], k=1)
        dist = hits[0][1]
        cos = 1.0 - dist / 2.0
        src = hits[0][0].metadata.get("source", "?")[:22]
    except Exception as e:
        cos, src = -1, str(e)
    tag = "域外/特殊" if s["source_doc"] in ("未命中兜底", "闲聊路由") or s["sid"] in ("44", "47", "61", "62") else "知识题"
    print(f"{sid} [{tag}] dense={cos:.3f} ({src})  {s['question']}")
