# -*- coding: utf-8 -*-
"""retrieval 模块单元测试（不依赖外部服务，纯函数测试）
运行：pytest tests/test_retrieval.py -v
"""
import hashlib
from pathlib import Path
from langchain_core.documents import Document

from core.retrieval import (
    format_docs,
    _doc_key,
    calculate_file_hash,
    safe_filename,
    resolve_rerank_model_path,
    tokenize_for_bm25,
    SPLIT_STRATEGIES,
    _split_recursive,
    do_retrieve,
)


# ---------- format_docs ----------
class TestFormatDocs:
    def test_empty(self):
        assert "没有检索到相关资料" in format_docs([])

    def test_single_doc(self):
        doc = Document(page_content="测试内容", metadata={"source": "test.md", "chunk_id": 1})
        result = format_docs([doc])
        assert "测试内容" in result
        assert "test.md" in result
        assert "1" in result

    def test_multiple_docs(self):
        docs = [
            Document(page_content="内容A", metadata={"source": "a.md", "chunk_id": 1}),
            Document(page_content="内容B", metadata={"source": "b.md", "chunk_id": 2}),
        ]
        result = format_docs(docs)
        assert "内容A" in result
        assert "内容B" in result
        assert "资料 1" in result
        assert "资料 2" in result


# ---------- _doc_key ----------
class TestDocKey:
    def test_with_hash_and_id(self):
        doc = Document(page_content="x", metadata={"file_hash": "abc123", "chunk_id": 5})
        assert _doc_key(doc) == "abc123_5"

    def test_fallback_to_content_hash(self):
        doc = Document(page_content="测试内容", metadata={})
        expected = hashlib.md5("测试内容".encode("utf-8")).hexdigest()
        assert _doc_key(doc) == expected


# ---------- calculate_file_hash ----------
class TestCalculateFileHash:
    def test_known_hash(self):
        data = b"hello"
        expected = hashlib.sha256(data).hexdigest()
        assert calculate_file_hash(data) == expected

    def test_different_inputs(self):
        assert calculate_file_hash(b"a") != calculate_file_hash(b"b")


# ---------- safe_filename ----------
class TestSafeFilename:
    def test_plain_name(self):
        assert safe_filename("doc.md") == "doc.md"

    def test_path_traversal(self):
        assert safe_filename("../../../etc/passwd") == "passwd"

    def test_windows_path(self):
        assert safe_filename("C:\\Users\\test\\file.pdf") == "file.pdf"


# ---------- resolve_rerank_model_path ----------
class TestResolveRerankPath:
    def test_known_model_returns_path_or_id(self):
        # 本地目录不存在时返回 HF ID；存在时返回本地路径
        result = resolve_rerank_model_path("BAAI/bge-reranker-base")
        assert "bge-reranker" in result

    def test_unknown_model_returns_id(self):
        assert resolve_rerank_model_path("unknown/model") == "unknown/model"


# ---------- tokenize_for_bm25 ----------
class TestTokenizeBM25:
    def test_chinese(self):
        tokens = tokenize_for_bm25("智能客服系统")
        assert len(tokens) > 0
        assert all(isinstance(t, str) for t in tokens)

    def test_english(self):
        tokens = tokenize_for_bm25("hello world")
        assert "hello" in tokens
        assert "world" in tokens

    def test_empty(self):
        assert tokenize_for_bm25("") == []

    def test_whitespace_only(self):
        assert tokenize_for_bm25("   ") == []


# ---------- SPLIT_STRATEGIES ----------
def test_split_strategies_complete():
    assert "QA对切分" in SPLIT_STRATEGIES
    assert "标题切分" in SPLIT_STRATEGIES
    assert "语义切分" in SPLIT_STRATEGIES
    assert "表格行切分" in SPLIT_STRATEGIES
    assert len(SPLIT_STRATEGIES) == 4


# ---------- _split_recursive ----------
class TestSplitRecursive:
    def test_short_text_single_chunk(self):
        docs = [Document(page_content="短文本", metadata={"source": "t.md"})]
        chunks = _split_recursive(docs, chunk_size=512, chunk_overlap=100)
        assert len(chunks) == 1
        assert chunks[0].page_content == "短文本"

    def test_long_text_splits(self):
        text = "这是测试内容。" * 100  # 足够长
        docs = [Document(page_content=text, metadata={"source": "t.md"})]
        chunks = _split_recursive(docs, chunk_size=100, chunk_overlap=20)
        assert len(chunks) > 1

    def test_preserves_metadata(self):
        docs = [Document(page_content="内容", metadata={"source": "custom.md"})]
        chunks = _split_recursive(docs, chunk_size=512, chunk_overlap=100)
        assert chunks[0].metadata["source"] == "custom.md"


# ---------- do_retrieve（空集合直接返回） ----------
class TestDoRetrieve:
    def test_empty_sources_returns_empty(self):
        """sources 为空集合时直接返回空列表（无关联文档）"""
        result = do_retrieve("test", 5, 0.0, "向量检索", sources=set())
        assert result == []

    def test_empty_sources_keyword(self):
        result = do_retrieve("test", 5, 0.0, "关键词检索", sources=set())
        assert result == []

    def test_empty_sources_hybrid(self):
        result = do_retrieve("test", 5, 0.0, "混合检索", sources=set())
        assert result == []
