# =========================================================
# LangGraph Agent 编排层
#   架构：START → [意图识别节点·Function Calling 路由] --条件边--> 4 条业务分支
#     · 知识库咨询：rag_retrieve（改写→检索→Rerank→置信度兜底）
#                   --条件边--> rag_answer（LLM 生成）/ END（兜底直答）
#     · 查订单：order 节点（工具调用）
#     · 建工单：ticket 节点（工具调用）
#     · 日常闲聊：chitchat 节点（LLM）
#   依赖通过 build_agent_graph(deps) 注入，避免与 appv1 循环导入。
#   流式：调用方使用 graph.stream(..., stream_mode=["messages", "values"])
#         messages 模式天然透传节点内 LLM 的逐 token 输出。
# =========================================================
from typing import Any, Callable, TypedDict
import re

from langgraph.graph import END, START, StateGraph

# 会产生面向用户文本的 LLM 节点（messages 流中只透传这些节点的 token）
ANSWER_NODES = {"rag_answer", "chitchat"}

# 双信号兜底门限：Rerank 低于阈值时，若向量 top1 余弦置信度 ≥ 此值则放行（实测真知识题≥0.63、域外题≤0.46）
DENSE_RESCUE_GATE = 0.55

# 确定性升级护栏：法律/监管威胁措辞属最高优先级投诉，即使 LLM 路由抖动也强制建工单 P1
ESCALATION_PATTERN = re.compile(r"法院|起诉|立案|12315|消协|监管|举报|报警|315投诉")


class AgentState(TypedDict, total=False):
    """图的共享状态：各节点读写这一个 State 对象"""

    # 输入
    question: str
    history: str
    top_k: int
    similarity_threshold: float
    retrieval_mode: str
    rerank_enabled: bool
    rerank_candidates: int
    confidence_fallback_enabled: bool
    confidence_threshold: float
    active_sources: set | None  # 关联知识库过滤：None=全库，集合=仅这些来源文件
    # 中间状态 / 输出
    intent: int
    intent_name: str
    tool_args: dict  # Function Calling：路由 LLM 提取的工具参数（order_id/issue/priority）
    route_note: str
    rewritten_q: str
    rewrite_ok: bool  # 问题改写是否成功（False=改写异常已降级为原问题，供链路评测诊断）
    docs: list
    answer: str
    trace: list  # 节点执行轨迹（供前端展示"图跑到了哪"）


def build_agent_graph(deps: dict[str, Any]):
    """根据注入的依赖编译并返回 LangGraph 可执行图。

    deps 需提供：
      route_intent(question, history) -> {"intent": int, "tool_args": dict}
          Function Calling 意图路由：LLM bind_tools 决定意图并提取工具参数
      run_order_query(question, tool_args) -> (answer, note)
      run_ticket_create(question, tool_args) -> (answer, note)
      rewrite_chain / answer_chain / chitchat_chain  (LangChain LCEL 链)
      do_retrieve(query, top_k, similarity_threshold, retrieval_mode, sources=None) -> docs
      rerank_docs(question, docs, top_n) -> docs
      format_docs(docs) -> str
      intent_names: dict[int, str]
      logger
    """
    route_intent: Callable = deps["route_intent"]
    run_order_query: Callable = deps["run_order_query"]
    run_ticket_create: Callable = deps["run_ticket_create"]
    rewrite_chain = deps["rewrite_chain"]
    answer_chain = deps["answer_chain"]
    chitchat_chain = deps["chitchat_chain"]
    do_retrieve: Callable = deps["do_retrieve"]
    rerank_docs: Callable = deps["rerank_docs"]
    format_docs: Callable = deps["format_docs"]
    intent_names: dict = deps["intent_names"]
    logger = deps["logger"]

    # ---------- 节点 1：意图识别（Function Calling 路由决策点） ----------
    def node_intent(state: AgentState) -> dict:
        # history 由图状态显式传入，节点不依赖 Streamlit 会话上下文
        decision = route_intent(state["question"], state.get("history", ""))
        intent = int(decision.get("intent", 1))
        tool_args = decision.get("tool_args") or {}
        guard = ""
        # 确定性护栏：LLM 误路由到 RAG/闲聊时，法律/监管威胁强制升级工单 P1
        if intent in (1, 4) and ESCALATION_PATTERN.search(state["question"]):
            intent = 3
            tool_args = {**tool_args, "priority": "P1"}
            guard = "（确定性升级护栏触发：法律/监管威胁→P1工单）"
            logger.warning(f"[Graph] 路由护栏强制升级 → 创建工单 question={state['question'][:50]}")
        logger.info(
            f"[Graph] 节点 intent(Function Calling) 完成 → {intent_names.get(intent, intent)} "
            f"tool_args={tool_args}{guard}"
        )
        trace = ["意图识别(Function Calling)"]
        if guard:
            trace.append("升级护栏")
        return {
            "intent": intent,
            "intent_name": intent_names.get(intent, "未知"),
            "tool_args": tool_args,
            "trace": trace,
        }

    # ---------- 节点 2a：查订单（Function Calling 工具执行分支） ----------
    def node_order(state: AgentState) -> dict:
        tool_args = state.get("tool_args") or {}
        answer, note = run_order_query(state["question"], tool_args)
        logger.info(f"[Graph] 节点 order 完成 → {note}")
        return {
            "answer": answer,
            "route_note": f"Function Calling 提取参数 {tool_args or '无'}；工具调用（查订单）：{note}",
            "docs": [],
            "trace": state.get("trace", []) + ["查订单工具"],
        }

    # ---------- 节点 2b：创建工单（Function Calling 工具执行分支） ----------
    def node_ticket(state: AgentState) -> dict:
        tool_args = state.get("tool_args") or {}
        answer, note = run_ticket_create(state["question"], tool_args)
        logger.info(f"[Graph] 节点 ticket 完成 → {note}")
        return {
            "answer": answer,
            "route_note": f"Function Calling 提取参数 {tool_args or '无'}；工具调用（建工单）：{note}",
            "docs": [],
            "trace": state.get("trace", []) + ["建工单工具"],
        }

    # ---------- 节点 2c：闲聊（LLM 分支，token 流式输出） ----------
    def node_chitchat(state: AgentState) -> dict:
        answer = chitchat_chain.invoke({"question": state["question"]})
        logger.info("[Graph] 节点 chitchat 完成")
        return {
            "answer": answer,
            "route_note": "直接回复",
            "docs": [],
            "trace": state.get("trace", []) + ["闲聊回复"],
        }

    # ---------- 节点 2d-①：RAG 检索准备（改写→检索→精排→置信度兜底） ----------
    def node_rag_retrieve(state: AgentState) -> dict:
        question = state["question"]
        trace = state.get("trace", []) + ["问题改写"]

        # 1) 问题改写（失败兜底为原问题，rewrite_ok=False 供链路评测归因）
        rewrite_ok = True
        try:
            rewritten = rewrite_chain.invoke(
                {"history": state.get("history", ""), "question": question}
            ).strip()
        except Exception as e:
            logger.error(f"[Graph] 问题改写失败，使用原问题: {e}")
            rewritten = question
            rewrite_ok = False
        trace.append("知识检索")

        # 2) 粗检索（开启 Rerank 时先多召回候选）
        rerank_on = state.get("rerank_enabled", False)
        top_k = state.get("top_k", 5)
        candidate_k = max(top_k, state.get("rerank_candidates", 10)) if rerank_on else top_k
        try:
            docs = do_retrieve(
                rewritten,
                top_k=candidate_k,
                similarity_threshold=state.get("similarity_threshold", 0.0),
                retrieval_mode=state.get("retrieval_mode", "向量检索"),
                sources=state.get("active_sources"),
            )
        except Exception as e:
            logger.exception(f"[Graph] 检索失败 question={question[:50]}: {e}")
            return {
                "rewritten_q": rewritten,
                "rewrite_ok": rewrite_ok,
                "docs": [],
                "answer": "检索失败，请稍后重试，或查看服务日志排查。",
                "route_note": f"{state.get('retrieval_mode', '向量检索')} → 检索异常",
                "trace": trace + ["检索异常"],
            }

        # 3) Rerank 精排 + 置信度兜底
        route_note = f"{state.get('retrieval_mode', '向量检索')} → RAG 链"
        if rerank_on and docs:
            docs = rerank_docs(rewritten, docs, top_k)
            trace.append("Rerank精排")
            # 精排降级（模型加载/推理失败，rerank_docs 返回原始排序并打 _rerank_degraded 标记）：
            # 此时没有真实相关性分数，严禁按缺失分 1.0 当满分放行，否则垃圾资料会绕过置信度兜底。
            rerank_degraded = bool(docs[0].metadata.get("_rerank_degraded"))
            if rerank_degraded:
                trace.append("精排降级")
                logger.warning(
                    f"[Graph] Rerank 精排未生效（已降级原始排序），改用向量置信度复核 "
                    f"question={question[:50]}"
                )
                # 置 0 分强制进入下方双信号门：向量强命中可放行，否则转人工兜底；全程留痕
                top_score = 0.0
                route_note += " → 精排降级(原始排序)"
            else:
                top_score = float(docs[0].metadata.get("rerank_score", 1.0)) if docs else 0.0
            if (
                docs
                and state.get("confidence_fallback_enabled", False)
                and top_score < state.get("confidence_threshold", 0.5)
            ):
                # 双信号兜底：Rerank 低分 + 向量路同样低置信度才转人工。
                # 防止口语化/错别字提问被 Reranker 低估而误杀（向量强命中即放行）。
                dense_fn = deps.get("dense_top_confidence")
                dense_top = float(dense_fn(rewritten, state.get("active_sources"))) if dense_fn else 0.0
                if dense_top >= DENSE_RESCUE_GATE:
                    logger.info(
                        f"[Graph] Rerank 低分但向量强命中，放行进入生成 | "
                        f"rerank={top_score:.3f} dense={dense_top:.3f} question={question[:50]}"
                    )
                    route_note += f" → 向量强命中放行({dense_top:.0%})"
                else:
                    threshold = state.get("confidence_threshold", 0.5)
                    logger.info(
                        f"[Graph] 置信度兜底触发 | rerank={top_score:.3f} dense={dense_top:.3f} "
                        f"阈值={threshold} question={question[:50]}"
                    )
                    return {
                        "rewritten_q": rewritten,
                        "rewrite_ok": rewrite_ok,
                        "docs": docs,
                        "answer": (
                            f"根据当前知识库资料，暂时没有找到与您问题高度相关的内容"
                            f"（最高相关性 {top_score:.0%}，低于置信度阈值 {threshold:.0%}）。\n\n"
                            f"建议您：\n"
                            f"- 更换问法或补充关键信息后重试\n"
                            f"- 如需进一步帮助，请联系人工客服"
                        ),
                        "route_note": route_note + " → 置信度兜底",
                        "trace": trace + ["置信度兜底"],
                    }

        # 4) 无检索结果：直接兜底文本
        if not docs:
            return {
                "rewritten_q": rewritten,
                "rewrite_ok": rewrite_ok,
                "docs": [],
                "answer": "未找到相关资料，请换个问法重试，或联系人工客服获取帮助。",
                "route_note": route_note + " → 无命中",
                "trace": trace + ["无命中兜底"],
            }

        return {
            "rewritten_q": rewritten,
            "rewrite_ok": rewrite_ok,
            "docs": docs,
            "route_note": route_note,
            "trace": trace,
        }

    # ---------- 节点 2d-②：RAG 答案生成（LLM，token 流式输出） ----------
    def node_rag_answer(state: AgentState) -> dict:
        context = format_docs(state["docs"])
        # 用 .stream() 逐 token 生成，配合 stream_mode="messages" 实现前端打字机效果
        answer = "".join(
            answer_chain.stream(
                {
                    "question": state["question"],
                    "rewritten_question": state.get("rewritten_q", state["question"]),
                    "context": context,
                }
            )
        )
        logger.info(
            f"[Graph] 节点 rag_answer 完成 | mode={state.get('retrieval_mode')} "
            f"rerank={state.get('rerank_enabled', False)} top_k={state.get('top_k')} "
            f"命中={len(state.get('docs', []))} question={state['question'][:50]}"
        )
        return {
            "answer": answer,
            "trace": state.get("trace", []) + ["生成回答"],
        }

    # ---------- 条件边：意图识别后的 4 选 1 路由 ----------
    def route_by_intent(state: AgentState) -> int:
        return state.get("intent", 1)

    # ---------- 条件边：检索后决定"生成回答"还是"直接结束（兜底文本）" ----------
    def route_after_retrieve(state: AgentState) -> str:
        return "end" if state.get("answer") else "answer"

    # ---------- 组装图 ----------
    graph = StateGraph(AgentState)
    graph.add_node("intent", node_intent)
    graph.add_node("order", node_order)
    graph.add_node("ticket", node_ticket)
    graph.add_node("chitchat", node_chitchat)
    graph.add_node("rag_retrieve", node_rag_retrieve)
    graph.add_node("rag_answer", node_rag_answer)

    graph.add_edge(START, "intent")
    graph.add_conditional_edges(
        "intent",
        route_by_intent,
        {1: "rag_retrieve", 2: "order", 3: "ticket", 4: "chitchat"},
    )
    graph.add_conditional_edges(
        "rag_retrieve",
        route_after_retrieve,
        {"answer": "rag_answer", "end": END},
    )
    graph.add_edge("rag_answer", END)
    graph.add_edge("order", END)
    graph.add_edge("ticket", END)
    graph.add_edge("chitchat", END)

    return graph.compile()
