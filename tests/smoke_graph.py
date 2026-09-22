# -*- coding: utf-8 -*-
"""Graph 冒烟测试：mock 全部依赖，验证四条路由 + 条件边 + 状态流转
运行：python tests/smoke_graph.py（需在项目根目录执行）
"""
import logging
import sys
from pathlib import Path

# 支持直接运行：把项目根加入 sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.agent_graph import build_agent_graph, ANSWER_NODES

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("smoke")

call_log = []


class FakeChain:
    def __init__(self, text):
        self.text = text

    def invoke(self, payload):
        call_log.append(("chain", self.text, str(payload)[:40]))
        return self.text

    def stream(self, payload):
        # 模拟真实 LCEL 链的逐 token 流式
        for ch in self.text:
            yield ch


fake_docs = []
captured_sources = []


def fake_do_retrieve(q, top_k, similarity_threshold, retrieval_mode, sources=None):
    captured_sources.append(sources)
    return fake_docs


deps = {
    "route_intent": lambda q, history="": {
        "intent": smoke_intent[0],
        "tool_args": {"order_id": "2024001", "issue": "耳机质量问题", "priority": "P1"},
    },
    "run_order_query": lambda q, tool_args=None: (f"订单回答:{q}", "query_order(2024001) → 成功"),
    "run_ticket_create": lambda q, tool_args=None: (f"工单回答:{q}", "create_ticket(priority=P1) → 成功"),
    "rewrite_chain": FakeChain("改写后的问题"),
    "answer_chain": FakeChain("这是RAG生成的最终答案。"),
    "chitchat_chain": FakeChain("闲聊回复～"),
    "do_retrieve": fake_do_retrieve,
    "rerank_docs": lambda q, docs, top_n: docs,
    "format_docs": lambda docs: "拼接后的上下文",
    "intent_names": {1: "知识库咨询", 2: "查订单", 3: "创建工单", 4: "日常闲聊"},
    "logger": logger,
}

graph = build_agent_graph(deps)

smoke_intent = [2]
r = graph.invoke({"question": "查订单 2024001", "top_k": 5, "retrieval_mode": "向量检索"})
assert "订单回答" in r["answer"], r
assert r["intent_name"] == "查订单"
assert "查订单工具" in r["trace"]
print("路由2(查订单) OK  trace:", " → ".join(r["trace"]))

smoke_intent[0] = 3
r = graph.invoke({"question": "我要投诉，耳机质量有严重问题必须马上处理", "top_k": 5})
assert "工单回答" in r["answer"]
assert "建工单工具" in r["trace"]
print("路由3(建工单) OK  trace:", " → ".join(r["trace"]))

smoke_intent[0] = 4
r = graph.invoke({"question": "你好", "top_k": 5})
assert r["answer"] == "闲聊回复～"
assert "闲聊回复" in r["trace"]
print("路由4(闲聊)   OK  trace:", " → ".join(r["trace"]))

# 路由1-RAG：有检索结果 → 走 rag_answer 生成
smoke_intent[0] = 1
from langchain_core.documents import Document
fake_docs.append(Document(page_content="测试资料内容", metadata={"source": "t.md", "chunk_id": 1}))
r = graph.invoke({"question": "退货政策", "top_k": 5, "retrieval_mode": "向量检索",
                  "rerank_enabled": False})
assert r["answer"] == "这是RAG生成的最终答案。"
assert r["rewritten_q"] == "改写后的问题"
assert len(r["docs"]) == 1
assert "生成回答" in r["trace"], r["trace"]
print("路由1(RAG生成) OK trace:", " → ".join(r["trace"]))

# 路由1-RAG：无检索结果 → 条件边走 END 兜底，不进 rag_answer
fake_docs.clear()
r = graph.invoke({"question": "无关问题", "top_k": 5, "retrieval_mode": "向量检索"})
assert "未找到相关资料" in r["answer"]
assert "生成回答" not in r["trace"]
print("路由1(无命中兜底) OK trace:", " → ".join(r["trace"]))

# 置信度兜底：rerank 开启且最高分低于阈值
fake_docs.append(Document(page_content="低相关资料", metadata={"source": "t.md", "chunk_id": 1, "rerank_score": 0.1}))
r = graph.invoke({"question": "偏题问题", "top_k": 5, "retrieval_mode": "向量检索",
                  "rerank_enabled": True, "confidence_fallback_enabled": True,
                  "confidence_threshold": 0.5, "rerank_candidates": 10})
assert "没有找到与您问题高度相关的内容" in r["answer"]
assert "置信度兜底" in r["trace"]
print("路由1(置信度兜底) OK trace:", " → ".join(r["trace"]))

# 来源过滤参数透传到 do_retrieve
graph.invoke({"question": "退货政策", "top_k": 5, "active_sources": {"售后政策.md"}})
assert captured_sources[-1] == {"售后政策.md"}, captured_sources[-1]
graph.invoke({"question": "退货政策", "top_k": 5})
assert captured_sources[-1] is None, captured_sources[-1]
print("来源过滤透传 OK（勾选集合 / 全库 None）")

# stream_mode 双模式事件校验：
# values 模式每节点返回状态；messages 模式仅对真实 ChatModel 产生 token
# （mock 链不是聊天模型，无 token 属正常；真实链路在浏览器中验证）
smoke_intent[0] = 4
events = list(graph.stream({"question": "你好", "top_k": 5}, stream_mode=["messages", "values"]))
for m, c in events:
    assert m in ("messages", "values"), m
values_states = [c for m, c in events if m == "values"]
assert any(s.get("intent") == 4 for s in values_states)
assert values_states[-1].get("answer") == "闲聊回复～"
print("stream_mode 事件结构 OK（values 状态数:", len(values_states), "）")

print("\n全部冒烟测试通过 ✅")
