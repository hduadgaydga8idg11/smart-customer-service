# -*- coding: utf-8 -*-
"""64题来源覆盖快检：每题向量 top5 是否包含 ground_truth 标注的 source_doc"""
import sys
sys.path[:0] = [".", "eval_runs"]
from langchain_chroma import Chroma
from core import retrieval as R
from core.model_factory import build_embeddings, load_model_config
from eval_lib import load_samples

emb = build_embeddings(load_model_config())
vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
samples = load_samples()

rag = [s for s in samples if s["source_doc"].endswith(".md")]
hit, miss = 0, []
for s in rag:
    docs = R.vector_search(s["question"], 5, 0.0, None, vs=vs)
    srcs = {d.metadata.get("source") for d in docs}
    if s["source_doc"] in srcs:
        hit += 1
    else:
        miss.append((s["sid"], s["question"], s["source_doc"], sorted(srcs)))
print(f"知识库题 {len(rag)} 道，向量 top5 命中标注文档：{hit}/{len(rag)} = {hit/len(rag):.0%}")
for sid, q, want, got in miss:
    print(f"  MISS {sid} {q} | 应={want} | 实={got}")
