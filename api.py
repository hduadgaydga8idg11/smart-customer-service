# -*- coding: utf-8 -*-
"""
智能客服助手 FastAPI 接口
提供 REST API 对外暴露 LangGraph Agent 的问答能力

启动：
  uvicorn api:app --host 0.0.0.0 --port 8000 --reload

接口文档：
  http://localhost:8000/docs  (Swagger UI)
"""
import hmac
import logging
import os
import re
import time
from contextlib import asynccontextmanager
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()  # .env 须在 core.model_factory 导入前加载（导入时读取 OLLAMA_BASE_URL 等）

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from pydantic import BaseModel

# ---------- 项目模块 ----------
from core.retrieval import (
    set_vectorstore, do_retrieve, rerank_docs, format_docs,
    get_kb_sources, set_rerank_model,
)
from core.database import (
    db_create_conversation, db_save_message, db_load_messages,
    db_list_conversations, db_delete_conversation, build_history_text,
)
from core.prompts import rewrite_prompt, answer_prompt, intent_prompt, route_prompt
from core.tools import ALL_TOOLS, create_ticket_record, query_order_status
from langchain_core.prompts import ChatPromptTemplate
from core.model_factory import (
    LOCAL_CHAT_MODEL, LOCAL_EMBEDDING_MODEL,
    CHAT_PRESETS, EMBEDDING_PRESETS,
    load_model_config, build_runtime_models, embedding_space, describe_source,
)

# ---------- 日志 ----------
from logging.handlers import RotatingFileHandler

PROJECT_ROOT = Path(__file__).resolve().parent
LOG_DIR = PROJECT_ROOT / "logs"
LOG_DIR.mkdir(exist_ok=True)
logger = logging.getLogger("rag_app")
if not logger.handlers:
    logger.setLevel(logging.INFO)
    _h = RotatingFileHandler(LOG_DIR / "app.log", maxBytes=5 * 1024 * 1024, backupCount=3, encoding="utf-8")
    _h.setFormatter(logging.Formatter("%(asctime)s [%(levelname)s] %(message)s"))
    logger.addHandler(_h)

# ---------- 模型配置（与 Streamlit 主程序共享 config.yaml + .env） ----------
DB_PATH = str(PROJECT_ROOT / "chroma_db")
INTENT_NAMES = {1: "知识库咨询", 2: "查订单", 3: "创建工单", 4: "日常闲聊"}

# ---------- 全局资源（启动时加载） ----------
_resources = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    """应用生命周期：启动时加载模型，关闭时清理"""
    from langchain_chroma import Chroma
    from langchain_core.output_parsers import StrOutputParser

    logger.info("正在加载模型资源...")
    cfg = load_model_config()

    # 嵌入空间守卫：不一致直接拒绝启动，避免线上服务静默产出错乱检索结果
    cur_space = embedding_space(cfg)
    kb_space = cfg.get("kb_embedding_space", LOCAL_EMBEDDING_MODEL)
    if cur_space != kb_space:
        raise RuntimeError(
            f"嵌入模型向量空间({cur_space})与知识库建库空间({kb_space})不一致，拒绝启动。"
            "请先在 Streamlit「模型设置」页一键重建知识库完成迁移，或恢复原嵌入配置。"
        )

    # 构建模型（API 配置无效时自动回退本地，warning 记入日志）
    chat_model, embeddings, effective_cfg, warning = build_runtime_models(cfg)
    if warning:
        logger.warning(warning)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    set_vectorstore(vectorstore)
    set_rerank_model(cfg.get("rerank_model"))
    logger.info(
        f"聊天模型：{describe_source(effective_cfg['chat'], CHAT_PRESETS, LOCAL_CHAT_MODEL)} | "
        f"嵌入模型：{describe_source(effective_cfg['embedding'], EMBEDDING_PRESETS, LOCAL_EMBEDDING_MODEL)}"
    )

    # 构建链
    rewrite_chain = rewrite_prompt | chat_model | StrOutputParser()
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    intent_chain = intent_prompt | chat_model | StrOutputParser()
    chitchat_prompt = ChatPromptTemplate.from_template(
        "你是客服助手，请友好简洁地回复：\n用户：{question}\n助手："
    )
    chitchat_chain = chitchat_prompt | chat_model | StrOutputParser()

    # Function Calling：把业务工具绑定到 LLM（工具定义见 core/tools.py）
    route_llm = chat_model.bind_tools(ALL_TOOLS)

    _resources.update({
        "chat_model": chat_model,
        "vectorstore": vectorstore,
        "cfg": effective_cfg,
        "rewrite_chain": rewrite_chain,
        "answer_chain": answer_chain,
        "intent_chain": intent_chain,
        "chitchat_chain": chitchat_chain,
        "route_llm": route_llm,
    })

    # 构建图
    from core.agent_graph import build_agent_graph
    deps = {
        "route_intent": route_intent,
        "run_order_query": run_order_query,
        "run_ticket_create": run_ticket_create,
        "rewrite_chain": rewrite_chain,
        "answer_chain": answer_chain,
        "chitchat_chain": chitchat_chain,
        "do_retrieve": do_retrieve,
        "rerank_docs": rerank_docs,
        "format_docs": format_docs,
        "intent_names": INTENT_NAMES,
        "logger": logger,
    }
    _resources["graph"] = build_agent_graph(deps)
    logger.info("模型资源加载完成，API 就绪")
    yield
    logger.info("API 关闭")


# ---------- 工具实现已迁移至 core/tools.py（@tool Function Calling 统一维护） ----------


def classify_intent(question: str, history_text: str | None = None) -> int:
    """意图分类（API 版，不依赖 Streamlit session_state）"""
    if history_text is None:
        history_text = ""
    try:
        raw = _resources["intent_chain"].invoke({"history": history_text, "question": question}).strip()
        match = re.search(r"[1-4]", raw)
        intent = int(match.group()) if match else 1
    except Exception as e:
        logger.error(f"意图识别失败，兜底为知识库咨询: {e}")
        intent = 1
    logger.info(f"意图识别 | intent={intent}({INTENT_NAMES[intent]}) question={question[:50]}")
    return intent


def route_intent(question: str, history_text: str = "") -> dict:
    """Function Calling 意图路由（API 版，无 Streamlit 会话依赖）；失败时降级为提示词分类"""
    try:
        ai_msg = _resources["route_llm"].invoke(
            route_prompt.invoke(
                {"history": history_text or "", "question": question}
            ).to_messages()
        )
        tool_calls = getattr(ai_msg, "tool_calls", None) or []
        if tool_calls:
            tc = tool_calls[0]
            intent = 2 if tc["name"] == "query_order" else 3
            tool_args = dict(tc["args"] or {})
        else:
            content = (ai_msg.content or "").strip().upper()
            intent, tool_args = (4, {}) if "CHAT" in content else (1, {})
    except Exception as e:
        logger.error(f"Function Calling 路由失败，降级为提示词分类: {e}")
        intent, tool_args = classify_intent(question, history_text), {}
    logger.info(
        f"意图路由(Function Calling) | intent={intent}({INTENT_NAMES[intent]}) "
        f"tool_args={tool_args} question={question[:50]}"
    )
    return {"intent": intent, "tool_args": tool_args}


def extract_order_id(text: str) -> str | None:
    """从文本中提取订单号（6~8 位数字）—— LLM 未提取到参数时的正则兜底"""
    match = re.search(r"\d{6,8}", text)
    return match.group() if match else None


def run_order_query(question: str, tool_args: dict | None = None) -> tuple[str, str]:
    """查订单工具流程（参数来自 Function Calling 提取）；LLM 缺失时正则兜底"""
    tool_args = tool_args or {}
    order_id = str(tool_args.get("order_id") or "").strip() or (extract_order_id(question) or "")
    if not order_id:
        return ("请提供订单号（6~8 位数字），我帮您查询订单状态。", "参数不足（order_id 缺失），主动澄清")
    logger.info(f"调用工具 query_order | order_id={order_id}")
    order = query_order_status(order_id)
    if not order:
        return (f"未找到订单号 {order_id} 的记录，请确认订单号是否正确。", f"query_order({order_id}) → 未找到")
    reply = (
        f"订单号 {order_id}：\n"
        f"- 商品：{order['item']}\n"
        f"- 状态：{order['status']}\n"
        f"- 快递：{order['carrier']}（{order['tracking_no']}）\n"
        f"- 预计：{order['eta']}\n\n"
        "（演示数据：以上订单信息为功能演示，非真实订单）"
    )
    return reply, f"query_order({order_id}) → 成功"


def run_ticket_create(question: str, tool_args: dict | None = None) -> tuple[str, str]:
    """创建工单工具流程（参数来自 Function Calling 提取）；缺失时兜底"""
    tool_args = tool_args or {}
    issue = str(tool_args.get("issue") or "").strip() or question.strip()
    priority = str(tool_args.get("priority") or "").strip().upper()
    if priority not in ("P1", "P2"):
        priority = "P2"
    logger.info(f"调用工具 create_ticket | priority={priority} issue={issue[:50]}")
    ticket = create_ticket_record(issue, priority=priority)
    reply = (
        f"已为您创建工单：\n"
        f"- 工单号：{ticket['ticket_id']}\n"
        f"- 问题描述：{ticket['issue'][:50]}\n"
        f"- 优先级：{ticket['priority']}\n"
        f"- SLA：{ticket['sla']}\n"
        f"我们会尽快安排人工跟进。\n\n"
        "（演示模式：工单未真实提交，仅供功能演示）"
    )
    return reply, f"create_ticket(priority={ticket['priority']}) → 成功"


# ---------- API 模型 ----------
class ChatRequest(BaseModel):
    question: str
    conv_id: str | None = None  # 会话标识（对外 token，防遍历）
    top_k: int = 5
    retrieval_mode: str = "向量检索"
    similarity_threshold: float = 0.0
    rerank_enabled: bool = False
    confidence_fallback_enabled: bool = True
    confidence_threshold: float = 0.5
    sources: list[str] | None = None  # 检索范围（None=全库）


class ChatResponse(BaseModel):
    conv_id: str
    answer: str
    intent: str
    trace: list[str]
    docs: list[dict]
    rewritten_question: str | None = None


class ConversationCreate(BaseModel):
    title: str = "新对话"


# ---------- FastAPI 应用 ----------
app = FastAPI(
    title="智能客服助手 API",
    description="基于 LangGraph + RAG 的智能客服 Agent",
    version="1.0.0",
    lifespan=lifespan,
)


# ---------- 访问令牌鉴权（防止公网未授权调用 / 会话遍历 / 刷量消耗）----------
# 令牌在 .env 中配置 API_ACCESS_TOKEN；调用方需在请求头带 X-API-Key。
# /api/health 放行（供监控/负载均衡探测）；其余接口未带正确令牌一律拒绝。
@app.middleware("http")
async def api_auth_middleware(request: Request, call_next):
    if request.url.path == "/api/health":
        return await call_next(request)
    expected = (os.environ.get("API_ACCESS_TOKEN") or "").strip()
    if not expected:
        # 未配置令牌：安全默认，拒绝所有业务请求，提示配置
        return JSONResponse(
            status_code=503,
            content={"detail": "服务未配置访问令牌（API_ACCESS_TOKEN），已拒绝访问"},
        )
    provided = (request.headers.get("X-API-Key") or "").strip()
    # compare_digest 恒定时间比较，防时序攻击逐字节猜测令牌
    if not provided or not hmac.compare_digest(provided.encode(), expected.encode()):
        return JSONResponse(status_code=401, content={"detail": "无效的访问令牌"})
    return await call_next(request)


@app.get("/api/health")
async def health():
    """健康检查：仅返回存活状态，不暴露模型来源/知识库空间等部署信息"""
    return {"status": "ok"}


@app.get("/api/kb/sources")
def kb_sources():
    """知识库来源文件列表"""
    sources = get_kb_sources()
    return {"sources": sources, "count": len(sources)}


@app.get("/api/conversations")
def list_conversations(limit: int = 20):
    """会话列表（返回对外 token，不暴露自增 id）"""
    convs = db_list_conversations(limit)
    return {"conversations": [{"token": c[0], "title": c[1], "created_at": c[2]} for c in convs]}


@app.post("/api/conversations", status_code=201)
def create_conversation(req: ConversationCreate):
    """创建新会话"""
    token = db_create_conversation(req.title)
    return {"token": token}


@app.get("/api/conversations/{conv_token}")
def get_conversation(conv_token: str):
    """获取会话历史消息（按对外 token）"""
    messages = db_load_messages(conv_token)
    return {"conv_token": conv_token, "messages": messages}


@app.delete("/api/conversations/{conv_token}")
def delete_conversation(conv_token: str):
    """删除会话（按对外 token）"""
    db_delete_conversation(conv_token)
    return {"deleted": conv_token}


@app.post("/api/chat", response_model=ChatResponse)
def chat(req: ChatRequest):
    """智能问答接口：接收问题，返回 Agent 回答"""
    _t_req = time.time()
    graph = _resources.get("graph")
    if graph is None:
        raise HTTPException(status_code=503, detail="Agent 尚未就绪，请稍后重试")

    # 会话管理
    if req.conv_id is None:
        req.conv_id = db_create_conversation(req.question[:20])

    # 加载历史
    history_messages = db_load_messages(req.conv_id)
    history_text = build_history_text(history_messages)

    # source 过滤
    active_sources = set(req.sources) if req.sources else None

    # 调用图
    initial_state = {
        "question": req.question,
        "history": history_text,  # 多轮上下文：意图识别与问题改写依赖（与主页一致）
        "top_k": req.top_k,
        "similarity_threshold": req.similarity_threshold,
        "retrieval_mode": req.retrieval_mode,
        "rerank_enabled": req.rerank_enabled,
        "rerank_candidates": 10,
        "confidence_fallback_enabled": req.confidence_fallback_enabled,
        "confidence_threshold": req.confidence_threshold,
        "active_sources": active_sources,
    }

    try:
        result = graph.invoke(initial_state)
    except Exception as e:
        # 详情只进服务端日志；响应给客户端的 detail 不携带异常原文，避免泄露服务器路径/内部信息
        logger.error(f"Agent 调用失败: {e}", exc_info=True)
        raise HTTPException(status_code=500, detail="Agent 处理失败，请稍后重试或查看服务日志")

    answer = result.get("answer", "")
    intent = result.get("intent_name", "")
    trace = result.get("trace", [])
    docs = result.get("docs", [])
    rewritten_q = result.get("rewritten_q")

    # 保存消息（与主页口径对齐：补齐耗时/命中数/token 字段，保证报表统计不缺数据）
    _t_api = time.time()
    db_save_message(req.conv_id, {"role": "user", "content": req.question})
    db_save_message(req.conv_id, {
        "role": "assistant",
        "content": answer,
        "intent_name": intent,
        "route_note": " → ".join(trace),
        "rewritten_question": rewritten_q,
        "retrieval_mode": req.retrieval_mode,
        "docs": docs,
        "total_time": round(_t_api - _t_req, 2),
        "doc_count": len(docs),
    })

    # 格式化文档返回
    docs_out = [{"content": d.page_content, "metadata": d.metadata} for d in docs]

    return ChatResponse(
        conv_id=req.conv_id,
        answer=answer,
        intent=intent,
        trace=trace,
        docs=docs_out,
        rewritten_question=rewritten_q,
    )


if __name__ == "__main__":
    import uvicorn
    uvicorn.run("api:app", host="0.0.0.0", port=8000, reload=True)
