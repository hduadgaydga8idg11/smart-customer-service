# -*- coding: utf-8 -*-
"""全链路节点级评测核心（框架无关层）

把"跑真实生产 LangGraph → 逐节点采集输出/耗时/状态 → LLM 裁判打分 → 错误归因"
的逻辑从 Streamlit 页面抽出，供两处复用：
  1. pages/03_评测系统.py 的「Agent 全链路评测」模块（本地看板）
  2. core/eval_langsmith.py（把同一份评分结果上报 LangSmith 实验看板）

依赖通过 deps 字典注入，不 import 任何 UI 框架：
  deps = {
      "graph": agent_graph,          # 编译后的 LangGraph 生产图
      "chat_model": chat_model,      # 裁判 LLM
      "embeddings": embeddings,      # 语义相似度用
  }
cfg = {
      "mode": "混合检索", "rerank": False, "rerank_k": 10,
      "fallback_on": True, "fallback_th": 0.5,
      "top_k": 5, "threshold": 0.0,
}
"""
import json
import logging
import re
import time

import numpy as np
from langchain_core.callbacks import BaseCallbackHandler

logger = logging.getLogger("rag_app")

# 期望意图的各种写法 → 标准意图名
INTENT_ALIASES = {
    "1": "知识库咨询", "知识库": "知识库咨询", "知识库咨询": "知识库咨询",
    "知识咨询": "知识库咨询", "rag": "知识库咨询",
    "2": "查订单", "查订单": "查订单", "订单": "查订单",
    "查物流": "查订单", "order": "查订单",
    "3": "创建工单", "建工单": "创建工单", "创建工单": "创建工单",
    "工单": "创建工单", "投诉": "创建工单", "ticket": "创建工单",
    "4": "日常闲聊", "闲聊": "日常闲聊", "日常闲聊": "日常闲聊",
    "chat": "日常闲聊", "chitchat": "日常闲聊",
}

# 图节点名 → 中文展示名
NODE_LABELS = {
    "intent": "🧭 路由节点",
    "rag_retrieve": "🔍 改写+检索节点",
    "rag_answer": "💬 回复节点(RAG)",
    "order": "🛒 工具节点(查订单)",
    "ticket": "🎫 工具节点(建工单)",
    "chitchat": "💬 回复节点(闲聊)",
}

STATUS_ICON = {"ok": "✅", "warn": "⚠️", "error": "❌", "pending": "❔", None: "—"}


def _extract_usage(response) -> dict | None:
    """从 LLM 结果对象提取 token 用量（优先 message.usage_metadata，兼容 llm_output）。
    部分供应商流式调用不返回 usage，此时返回 None（页面显示 —）。"""
    try:
        gens = getattr(response, "generations", None) or []
        if gens and gens[0]:
            msg = getattr(gens[0][0], "message", None)
            meta = getattr(msg, "usage_metadata", None)
            if meta:
                return meta
        llm_out = getattr(response, "llm_output", None) or {}
        tu = llm_out.get("token_usage") or {}
        if tu:
            return {
                "input_tokens": tu.get("prompt_tokens", 0) or 0,
                "output_tokens": tu.get("completion_tokens", 0) or 0,
                "total_tokens": tu.get("total_tokens", 0) or 0,
            }
    except Exception:
        pass
    return None


class _NodeTokenCollector(BaseCallbackHandler):
    """按图节点归集 LLM token 用量（零侵入生产图）。

    原理：graph.stream(config={"callbacks": [...]}) 会把回调传播到节点内的所有
    LLM 调用；LangGraph 节点执行时 run 名即节点名，on_chain_start 捕获当前节点，
    on_llm_end 把该次调用的 usage 记到当前节点名下。图按路径线性执行，
    节点间串行，用 current 游标即可正确归属。
    注意：评测裁判（judge_chain）的 token 不在采集范围（仅统计业务链路消耗）。
    """

    NODE_NAMES = {"intent", "rag_retrieve", "rag_answer", "order", "ticket", "chitchat"}

    def __init__(self):
        self.usage_by_node: dict[str, dict] = {}
        self._current: str | None = None

    def on_chain_start(self, serialized, inputs, *, run_id=None, parent_run_id=None,
                       name=None, **kwargs):
        nm = name or (serialized or {}).get("name", "")
        if nm in self.NODE_NAMES:
            self._current = nm
            self.usage_by_node.setdefault(
                nm, {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "calls": 0}
            )

    def on_chain_end(self, outputs, *, run_id=None, parent_run_id=None, name=None, **kwargs):
        nm = name or ""
        if nm in self.NODE_NAMES and self._current == nm:
            self._current = None

    def on_llm_end(self, response, *, run_id=None, parent_run_id=None, **kwargs):
        if self._current is None:
            return
        usage = _extract_usage(response)
        if usage:
            bucket = self.usage_by_node[self._current]
            bucket["input_tokens"] += int(usage.get("input_tokens", 0) or 0)
            bucket["output_tokens"] += int(usage.get("output_tokens", 0) or 0)
            bucket["total_tokens"] += int(usage.get("total_tokens", 0) or 0)
            bucket["calls"] += 1


def normalize_intent(raw: str) -> str:
    """期望意图归一化（支持中文名/数字/英文别名），无法识别返回空串"""
    if raw is None or not str(raw).strip():
        return ""
    text = str(raw).strip()
    return INTENT_ALIASES.get(text.lower(), INTENT_ALIASES.get(text, ""))


def compute_semantic_similarity(embeddings, text1: str, text2: str) -> float:
    """两段文本的 embedding 余弦相似度（与标准答案比对用）"""
    try:
        t1, t2 = text1[:1000], text2[:1000]
        emb1 = np.array(embeddings.embed_query(t1))
        emb2 = np.array(embeddings.embed_query(t2))
        return float(np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2)))
    except Exception:
        return 0.0


def judge_chain(chat_model, question: str, final_state: dict, node_names: list) -> dict:
    """LLM 裁判：按实际执行的分支动态构造评分项，对各节点输出打分（0-10）。
    评分失败不阻塞评测，但显式返回 {"judge_failed": True}（调用方据此标记，
    不得把空评分当满分聚合）。"""
    scores: dict = {}
    try:
        intent_name = final_state.get("intent_name", "未知")
        tool_args = final_state.get("tool_args") or {}
        docs = final_state.get("docs", [])
        lines = [
            "你是客服 Agent 链路评测专家。请对以下 Agent 执行结果逐节点评分"
            "（0-10 分，10 分最好，严格打分）。",
            "",
            f"【用户问题】{question}",
            f"【路由节点输出】意图={intent_name}；提取的工具参数={tool_args or '无'}",
        ]
        score_items = [
            ("route_score",
             "路由合理性：意图识别是否正确（知识库咨询/查订单/创建工单/日常闲聊），"
             "工具参数提取是否正确"),
        ]
        if "rag_retrieve" in node_names:
            lines.append(f"【改写节点输出】改写后检索问题：{final_state.get('rewritten_q', '')}")
            lines.append("【检索节点输出】检索到的文档摘要：")
            if docs:
                for d in docs[:3]:
                    lines.append(f"- {d.page_content[:150]}")
            else:
                lines.append("- （无检索结果，触发了兜底）")
            score_items += [
                ("rewrite_score",
                 "改写质量：改写后问题是否保留核心语义且更适合检索；改写失败或改变原意给低分"),
                ("retrieval_score",
                 "检索相关性：检索文档是否包含回答问题所需的核心信息；无结果给 0 分"),
                ("faithfulness",
                 "答案忠实度：答案是否完全基于检索文档，无编造、无超出文档的承诺"),
            ]
        if "order" in node_names or "ticket" in node_names:
            score_items.append(
                ("tool_score",
                 "工具执行合理性：参数提取/澄清/执行结果是否合理"
                 "（参数不足时主动澄清是合理行为，应给高分）")
            )
        lines.append(f"【回复节点输出】{(final_state.get('answer') or '')[:800]}")
        score_items.append(
            ("relevance", "回复相关性：最终回复是否切中用户问题、真正解决用户需求")
        )
        lines += [
            "",
            "只输出 JSON，不要输出其他内容：{"
            + ", ".join(f'"{k}": 分数' for k, _ in score_items) + "}",
            "评分项说明：",
        ]
        for i, (_, desc) in enumerate(score_items, 1):
            lines.append(f"{i}. {desc}")

        resp = chat_model.invoke("\n".join(lines))
        json_match = re.search(r"\{.*\}", resp.content, re.DOTALL)
        if json_match:
            data = json.loads(json_match.group())
            for k, _ in score_items:
                try:
                    scores[k] = round(float(data.get(k, 0)), 1)
                except (TypeError, ValueError):
                    scores[k] = 0.0
    except Exception as e:
        logger.warning(f"LLM 裁判评分失败，显式标记 judge_failed question={question[:50]}: {e}")
        return {"judge_failed": True}
    return scores


def run_chain_eval_single(deps: dict, question: str, expected_intent: str,
                          ground_truth: str, cfg: dict) -> dict:
    """以真实生产图执行单题，stream_mode='updates' 逐节点采集输出与耗时，
    并完成节点状态判定、LLM 裁判评分、错误节点归因。

    deps: {"graph", "chat_model", "embeddings"}；cfg 见模块头注释。
    """
    graph = deps["graph"]
    chat_model = deps["chat_model"]
    embeddings = deps["embeddings"]

    initial_state = {
        "question": question,
        "history": "",  # 离线评测逐题独立，不带历史
        "top_k": cfg.get("top_k", 5),
        "similarity_threshold": cfg.get("threshold", 0.0),
        "retrieval_mode": cfg["mode"],
        "rerank_enabled": cfg["rerank"],
        "rerank_candidates": cfg["rerank_k"],
        "confidence_fallback_enabled": cfg["fallback_on"],
        "confidence_threshold": cfg["fallback_th"],
        "active_sources": None,  # 全库评测
    }

    # ---- 1) 跑图，逐节点采集（updates 模式：每个节点完成后产出 {节点名: 状态增量}）----
    # 回调经 config 传播到节点内所有 LLM 调用，按节点归集 token 用量（零侵入生产图）
    token_collector = _NodeTokenCollector()
    node_runs: list[dict] = []
    final_state = dict(initial_state)
    graph_error = None
    t_start = time.time()
    last_t = t_start
    try:
        for chunk in graph.stream(
            initial_state, stream_mode="updates",
            config={"callbacks": [token_collector]},
        ):
            now = time.time()
            for node_name, update in (chunk or {}).items():
                node_runs.append({
                    "node": node_name,
                    "latency": round(now - last_t, 3),
                    "update": update or {},
                })
                final_state.update(update or {})
                last_t = now
    except Exception as e:
        graph_error = str(e)
    total_time = round(time.time() - t_start, 3)

    node_names = [n["node"] for n in node_runs]
    intent_name = final_state.get("intent_name", "未识别")
    route_note = final_state.get("route_note", "")
    docs = final_state.get("docs", []) or []
    answer = final_state.get("answer", "") or ""

    # ---- 2) 节点状态判定（硬规则，不依赖 LLM）----
    # 路由节点：有期望意图 → 精确比对；无标注 → 待裁判评分
    exp_norm = normalize_intent(expected_intent)
    if exp_norm:
        route_status = "ok" if intent_name == exp_norm else "error"
        route_status_text = f"期望：{exp_norm}｜实际：{intent_name}"
    else:
        route_status = "pending"
        route_status_text = f"实际：{intent_name}（未提供期望意图，由 LLM 裁判评分）"

    # 改写节点（仅 RAG 分支经过）
    rewrite_status = None
    if "rag_retrieve" in node_names:
        rewrite_status = "ok" if final_state.get("rewrite_ok", True) else "error"

    # 检索节点
    retrieve_status = None
    retrieve_detail = ""
    # 精排降级时文档无真实 rerank_score（默认 0，严禁当满分），单独取标记用于状态判定
    rerank_degraded = bool(docs and docs[0].metadata.get("_rerank_degraded"))
    top_score = float(docs[0].metadata.get("rerank_score", 0)) if docs else 0.0
    if "rag_retrieve" in node_names:
        if "检索异常" in route_note:
            retrieve_status, retrieve_detail = "error", "检索服务异常"
        elif "无命中" in route_note:
            retrieve_status, retrieve_detail = "error", "知识库无命中"
        elif rerank_degraded:
            retrieve_status, retrieve_detail = "warn", "精排服务不可用，已降级原始排序（rerank 分数不可信）"
        elif "置信度兜底" in route_note:
            retrieve_status, retrieve_detail = "warn", f"置信度兜底（最高相关性 {top_score:.0%}）"
        else:
            retrieve_status = "ok"
            retrieve_detail = f"命中 {len(docs)} 块" + (
                f"，最高 rerank {top_score:.2f}" if top_score else ""
            )

    # 工具节点
    tool_status = None
    tool_detail = ""
    if "order" in node_names or "ticket" in node_names:
        if "澄清" in route_note:
            tool_status, tool_detail = "warn", "参数不足，主动澄清"
        elif "未找到" in route_note:
            tool_status, tool_detail = "warn", "工具未查到记录"
        else:
            tool_status, tool_detail = "ok", "工具执行成功"

    # 回复节点
    answer_status = "ok" if (answer and not graph_error) else ("error" if graph_error else None)

    # ---- 3) LLM 裁判逐节点评分 ----
    scores = judge_chain(chat_model, question, final_state, node_names)
    judge_failed = bool(scores.get("judge_failed"))
    if judge_failed:
        # 裁判失败必须显式标记：不得把缺省分当满分（否则评测虚高、归因失真）
        if route_status == "pending":
            route_status = "judge_failed"
    elif route_status == "pending":
        # 无标注时以裁判路由分 >=6 视为合理
        route_status = "ok" if scores.get("route_score", 0) >= 6 else "error"

    # 与标准答案的语义相似度
    sem_sim = 0.0
    if ground_truth and ground_truth.strip() and answer:
        sem_sim = round(compute_semantic_similarity(embeddings, answer, ground_truth), 3)

    # ---- 4) 错误归因：按因果顺序定位问题出在哪个节点 ----
    attribution = "✅ 正常"
    if graph_error:
        attribution = "图执行异常"
    elif judge_failed:
        attribution = "评测：裁判评分失败（未计入质量分）"
    elif route_status == "error":
        attribution = "路由节点：意图识别错误"
    elif retrieve_status == "error" and "异常" in retrieve_detail:
        attribution = "检索节点：检索服务异常"
    elif rewrite_status == "error":
        attribution = "改写节点：改写失败已降级原问题"
    elif retrieve_status == "error" and "无命中" in retrieve_detail:
        attribution = "检索节点：知识库无命中"
    elif retrieve_status == "warn" and "精排" in retrieve_detail:
        attribution = "检索节点：精排服务降级（rerank 未生效，分数不可信）"
    elif retrieve_status == "warn":
        attribution = "检索节点：置信度兜底"
    elif tool_status == "warn" and "澄清" in tool_detail:
        attribution = "工具节点：参数不足主动澄清"
    elif tool_status == "warn":
        attribution = "工具节点：未查到记录"
    elif scores.get("faithfulness", 10) < 6 or scores.get("relevance", 10) < 6:
        attribution = "回复节点：生成质量差"
    elif scores.get("retrieval_score", 10) < 6:
        attribution = "检索节点：召回内容不相关"
    elif scores.get("rewrite_score", 10) < 6 and "rag_retrieve" in node_names:
        attribution = "改写节点：改写质量差"

    return {
        "question": question,
        "expected_intent": exp_norm or "(未提供)",
        "predicted_intent": intent_name,
        "route_status": route_status,
        "route_status_text": route_status_text,
        "tool_args": final_state.get("tool_args") or {},
        "rewritten": final_state.get("rewritten_q", ""),
        "rewrite_status": rewrite_status,
        "retrieve_status": retrieve_status,
        "retrieve_detail": retrieve_detail,
        "tool_status": tool_status,
        "tool_detail": tool_detail,
        "answer_status": answer_status,
        "docs": docs,
        "answer": answer,
        "route_note": route_note,
        "node_runs": node_runs,
        "node_names": node_names,
        "scores": scores,
        "judge_failed": judge_failed,
        "semantic_similarity": sem_sim,
        "total_time": total_time,
        "attribution": attribution,
        "graph_error": graph_error,
        "llm_usage": token_collector.usage_by_node,
    }
