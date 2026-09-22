# -*- coding: utf-8 -*-
"""一次性数据修复：
1) 将新起草的退换货 FAQ 以生产流程（QA对切分 512/100）入库
2) 仅删除某品牌活动 xlsx 的向量块（保留 docs 原始文件）
"""
import shutil
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, ".")
from dotenv import load_dotenv
load_dotenv()

from langchain_chroma import Chroma
from core.model_factory import build_embeddings, load_model_config
from core.retrieval import (
    add_prepared_chunks_to_vectorstore,
    calculate_file_hash,
    get_all_file_records,
    set_vectorstore,
    split_file_for_preview,
)

DOCS = Path("docs")
emb = build_embeddings(load_model_config())
vs = Chroma(persist_directory="chroma_db", embedding_function=emb)
set_vectorstore(vs)

# ---------- 1) 入库退换货 FAQ ----------
raw = DOCS / "客服FAQ-退换货与售后政策.md"
data = raw.read_bytes()
fh = calculate_file_hash(data)
stored = DOCS / f"{fh[:12]}_客服FAQ-退换货与售后政策.md"
if not stored.exists():
    shutil.move(str(raw), stored)

chunks = split_file_for_preview(stored, 512, 100, "QA对切分", embeddings=emb)
print(f"切分块数: {len(chunks)}")
ok, msg = add_prepared_chunks_to_vectorstore(
    chunks, stored, "客服FAQ-退换货与售后政策.md", fh, vs=vs
)
print("退换货文档入库:", ok, msg)

# ---------- 2) 删除 xlsx 向量（保留文件） ----------
records = get_all_file_records(vs)
xlsx = [r for r in records.values() if r["source"].lower().endswith((".xlsx", ".xls"))]
for r in xlsx:
    print("删除向量（保留原文件）:", r["source"], r["chunk_count"], "块")
    vs.delete(where={"file_hash": r["file_hash"]})

# ---------- 3) 核对 ----------
data_all = vs.get(include=["metadatas"])
c = Counter(m.get("source", "?") for m in data_all["metadatas"])
print("\n=== 修复后 chroma_db 来源分布 ===")
for k, v in sorted(c.items()):
    print(f"{v:4d}  {k}")
print("总块数:", sum(c.values()))
