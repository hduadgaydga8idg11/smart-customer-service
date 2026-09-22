# -*- coding: utf-8 -*-
import sys
sys.path[:0] = [".", "eval_runs"]
from langchain_chroma import Chroma
from core import retrieval as R
from core.model_factory import build_embeddings, load_model_config
from core.retrieval import build_bm25_index_for_vectorstore

emb = build_embeddings(load_model_config())
vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
bm = build_bm25_index_for_vectorstore(vs)
q = "怎么申请退货？"

print("=== 向量 top10 ===")
vd = R.vector_search(q, 10, 0.0, None, vs=vs)
for i, d in enumerate(vd):
    print(f"[{i}]", d.metadata.get("source", "?")[:30], "|", d.page_content[:50].replace("\n", " "))

print("=== 关键词 top10 ===")
kd = R.keyword_search(q, 10, None, vs=vs, bm25_data=bm)
for i, d in enumerate(kd):
    print(f"[{i}]", d.metadata.get("source", "?")[:30], "|", d.page_content[:50].replace("\n", " "))

print("=== 全库含'申请退货'的块 ===")
res = vs.get(include=["documents", "metadatas"])
cnt = 0
for doc, meta in zip(res["documents"], res["metadatas"]):
    if "申请退货" in doc:
        cnt += 1
        print(" *", meta.get("source", "?")[:30], "|", doc[:70].replace("\n", " "))
print("含申请退货的块数:", cnt, "| 全库总块数:", len(res["documents"]))
