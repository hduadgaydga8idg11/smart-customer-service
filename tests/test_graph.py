# -*- coding: utf-8 -*-
"""pytest 版图路由冒烟测试（从 _smoke_graph.py 迁移）
运行：pytest tests/test_graph.py -v
"""
import pytest
from langchain_core.documents import Document
from core.agent_graph import build_agent_graph, ANSWER_NODES


# ---------- mock 依赖 ----------
class FakeChain:
    def __init__(self, text):
        self.text = text

    def invoke(self, payload):
        return self.text

    def stream(self, payload):
        for ch in self.text:
            yield ch


@pytest.fixture
def fake_deps(logger):
    """构造 mock 依赖字典"""
    return {
        "route_intent": lambda q, history="": {
            "intent": _smoke_intent[0],
            "tool_args": {"order_id": "2024001", "issue": "耳机质量问题", "priority": "P1"},
        },
        "run_order_query": lambda q, tool_args=None: (f"订单回答:{q}", "query_order(2024001) → 成功"),
        "run_ticket_create": lambda q, tool_args=None: (f"工单回答:{q}", "create_ticket(priority=P1) → 成功"),
        "rewrite_chain": FakeChain("改写后的问题"),
        "answer_chain": FakeChain("这是RAG生成的最终答案。"),
        "chitchat_chain": FakeChain("闲聊回复～"),
        "do_retrieve": lambda q, top_k=5, similarity_threshold=0.0, retrieval_mode="向量检索", sources=None: _fake_docs.copy(),
        "rerank_docs": lambda q, docs, top_n: docs,
        "format_docs": lambda docs: "拼接后的上下文",
        "intent_names": {1: "知识库咨询", 2: "查订单", 3: "创建工单", 4: "日常闲聊"},
        "logger": logger,
    }


_smoke_intent = [1]
_fake_docs = []


@pytest.fixture(autouse=True)
def _reset():
    _smoke_intent[0] = 1
    _fake_docs.clear()
    yield


# ---------- 路由2：查订单 ----------
def test_route_order(fake_deps):
    _smoke_intent[0] = 2
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "查订单 2024001", "top_k": 5, "retrieval_mode": "向量检索"})
    assert "订单回答" in r["answer"]
    assert r["intent_name"] == "查订单"
    assert "查订单工具" in r["trace"]


# ---------- 路由3：建工单 ----------
def test_route_ticket(fake_deps):
    _smoke_intent[0] = 3
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "我要投诉，耳机质量有严重问题必须马上处理", "top_k": 5})
    assert "工单回答" in r["answer"]
    assert "建工单工具" in r["trace"]


# ---------- Function Calling 参数透传 ----------
def test_tool_args_passthrough(fake_deps):
    """route_intent 提取的 tool_args 应透传到工具节点并展示在 route_note"""
    captured = {}

    def fake_order(q, tool_args=None):
        captured["args"] = tool_args
        return f"订单回答:{q}", "query_order(2024001) → 成功"

    fake_deps["run_order_query"] = fake_order
    _smoke_intent[0] = 2
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "查订单 2024001", "top_k": 5})
    assert captured["args"] == {"order_id": "2024001", "issue": "耳机质量问题", "priority": "P1"}
    assert "Function Calling 提取参数" in r["route_note"]


# ---------- 路由4：闲聊 ----------
def test_route_chitchat(fake_deps):
    _smoke_intent[0] = 4
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "你好", "top_k": 5})
    assert r["answer"] == "闲聊回复～"
    assert "闲聊回复" in r["trace"]


# ---------- 路由1-RAG：有检索结果 → 生成回答 ----------
def test_route_rag_generate(fake_deps):
    _smoke_intent[0] = 1
    _fake_docs.append(Document(page_content="测试资料内容", metadata={"source": "t.md", "chunk_id": 1}))
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "退货政策", "top_k": 5, "retrieval_mode": "向量检索",
                      "rerank_enabled": False})
    assert r["answer"] == "这是RAG生成的最终答案。"
    assert r["rewritten_q"] == "改写后的问题"
    assert len(r["docs"]) == 1
    assert "生成回答" in r["trace"]


# ---------- 路由1-RAG：无检索结果 → 无命中兜底 ----------
def test_route_rag_no_hit(fake_deps):
    _smoke_intent[0] = 1
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "无关问题", "top_k": 5, "retrieval_mode": "向量检索"})
    assert "未找到相关资料" in r["answer"]
    assert "生成回答" not in r["trace"]


# ---------- 路由1-RAG：置信度兜底 ----------
def test_route_confidence_fallback(fake_deps):
    _smoke_intent[0] = 1
    _fake_docs.append(Document(
        page_content="低相关资料",
        metadata={"source": "t.md", "chunk_id": 1, "rerank_score": 0.1},
    ))
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "偏题问题", "top_k": 5, "retrieval_mode": "向量检索",
                      "rerank_enabled": True, "confidence_fallback_enabled": True,
                      "confidence_threshold": 0.5, "rerank_candidates": 10})
    assert "没有找到与您问题高度相关的内容" in r["answer"]
    assert "置信度兜底" in r["trace"]


def test_route_confidence_dual_signal_rescue(fake_deps):
    """双信号兜底：Rerank 低分但向量强命中（口语题）时应放行进入生成，不转人工"""
    _smoke_intent[0] = 1
    _fake_docs.append(Document(
        page_content="发票开具相关资料",
        metadata={"source": "t.md", "chunk_id": 1, "rerank_score": 0.26},
    ))
    fake_deps["dense_top_confidence"] = lambda q, sources=None: 0.65
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "发pi票咋弄", "top_k": 5, "retrieval_mode": "混合检索",
                      "rerank_enabled": True, "confidence_fallback_enabled": True,
                      "confidence_threshold": 0.4, "rerank_candidates": 10})
    assert r["answer"] == "这是RAG生成的最终答案。"
    assert "置信度兜底" not in r["trace"]
    assert "向量强命中放行" in r["route_note"]


# ---------- 确定性升级护栏：法律/监管威胁强制 P1 工单 ----------
def test_escalation_guard_overrides_rag(fake_deps):
    """LLM 误路由到知识库咨询，但出现法院/举报等措辞 → 强制建工单 P1"""
    _smoke_intent[0] = 1
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "如果你们不退我就要去法院告你们！", "top_k": 5})
    assert r["intent_name"] == "创建工单"
    assert "工单回答" in r["answer"]
    assert "升级护栏" in r["trace"]
    assert r["tool_args"].get("priority") == "P1"


def test_escalation_guard_not_triggered_for_normal_question(fake_deps):
    """普通知识问题即使路由为 RAG，也不应被护栏拦截"""
    _smoke_intent[0] = 1
    _fake_docs.append(Document(page_content="测试资料", metadata={"source": "t.md", "chunk_id": 1}))
    graph = build_agent_graph(fake_deps)
    r = graph.invoke({"question": "退货政策是什么", "top_k": 5, "retrieval_mode": "向量检索",
                      "rerank_enabled": False})
    assert r["intent_name"] == "知识库咨询"
    assert "升级护栏" not in r["trace"]


# ---------- 来源过滤参数透传 ----------
def test_source_filter_passthrough(fake_deps):
    """active_sources 集合应透传到 do_retrieve；None 表示全库"""
    _smoke_intent[0] = 1
    captured = []

    def capture_retrieve(q, top_k=5, similarity_threshold=0.0, retrieval_mode="向量检索", sources=None):
        captured.append(sources)
        return _fake_docs.copy()

    fake_deps["do_retrieve"] = capture_retrieve
    graph = build_agent_graph(fake_deps)

    graph.invoke({"question": "退货政策", "top_k": 5, "active_sources": {"售后政策.md"}})
    assert captured[-1] == {"售后政策.md"}

    graph.invoke({"question": "退货政策", "top_k": 5})
    assert captured[-1] is None


# ---------- stream_mode 双模式事件结构 ----------
def test_stream_mode_events(fake_deps):
    _smoke_intent[0] = 4
    graph = build_agent_graph(fake_deps)
    events = list(graph.stream({"question": "你好", "top_k": 5}, stream_mode=["messages", "values"]))
    for m, _ in events:
        assert m in ("messages", "values")
    values_states = [c for m, c in events if m == "values"]
    assert any(s.get("intent") == 4 for s in values_states)
    assert values_states[-1].get("answer") == "闲聊回复～"
