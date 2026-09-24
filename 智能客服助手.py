# =========================================================
# 标准库导入
# =========================================================
import logging
import os
import re
import time
from datetime import datetime
from pathlib import Path

# =========================================================
# 加载 .env 环境变量（必须最先执行：core.model_factory 导入时即读取
# OLLAMA_BASE_URL 等环境变量，API Key 也依赖 .env 注入）
# =========================================================
from dotenv import load_dotenv

load_dotenv()  # 读取项目根目录下的 .env 文件

# =========================================================
# 第三方库导入（需提前安装）
# =========================================================
import numpy as np
import pandas as pd
import streamlit as st
from PIL import Image

# LangChain 相关
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate

# =========================================================
# 核心业务模块（core/ 包）
# =========================================================
from core.retrieval import (
    set_vectorstore,
    get_active_sources,
    do_retrieve, rerank_docs, format_docs, dense_top_confidence,
    set_rerank_model, get_rerank_last_error,
)
from core.model_factory import (
    LOCAL_EMBEDDING_MODEL,
    load_model_config, build_runtime_models,
    embedding_space,
)
from core.database import (
    CHAT_DB_PATH, db_execute, db_create_conversation, db_save_message,
    db_list_conversations, db_load_messages, db_update_feedback,
    db_delete_conversation, db_delete_message, db_export_all_messages,
    build_history_text,
)
from core.prompts import rewrite_prompt, answer_prompt, intent_prompt, route_prompt
from core.tools import ALL_TOOLS, MOCK_ORDERS, create_ticket_record, query_order_status
from core.chain_eval import _NodeTokenCollector  # 按节点归集 LLM token（零侵入回调）

# =========================================================
# 日志系统（生产规范：写入文件、按大小轮转、分级记录）
# =========================================================
from logging.handlers import RotatingFileHandler

LOG_DIR = Path(__file__).parent.resolve() / "logs"
LOG_DIR.mkdir(exist_ok=True)

logger = logging.getLogger("rag_app")
if not logger.handlers:  # Streamlit 每次交互都会重跑脚本，防止重复添加 handler
    logger.setLevel(logging.INFO)
    _file_handler = RotatingFileHandler(
        LOG_DIR / "app.log",
        maxBytes=5 * 1024 * 1024,  # 单文件最大 5MB
        backupCount=3,  # 最多保留 3 个历史文件
        encoding="utf-8",
    )
    _file_handler.setFormatter(
        logging.Formatter("%(asctime)s [%(levelname)s] %(message)s")
    )
    logger.addHandler(_file_handler)

# =========================================================
# 屏蔽冗余 TensorFlow 警告
# =========================================================
logging.getLogger("tensorflow").setLevel(logging.ERROR)

# =========================================================
# 项目根目录（用于所有相对路径）
# =========================================================
PROJECT_ROOT = Path(__file__).parent.resolve()
DB_PATH = str(PROJECT_ROOT / "chroma_db")  # 使用项目根目录下的绝对路径

# =========================================================
# 子页面统一外观（pages/ 各页 import 后显式调用 inject_subpage_style）
# 主页的 <style> 写在 main() 内、子页面 import 复用时执行不到；
# 公共规则集中于此：隐藏 Streamlit 默认菜单/页脚/Deploy 入口，背景色与主页一致
# =========================================================
SUBPAGE_CSS = """
<style>
#MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }
.stApp { background-color: #F8F9FA; }
</style>
"""


def inject_subpage_style() -> None:
    """子页面注入统一外观：隐藏默认菜单/Deploy/页脚，背景色与主页对齐。"""
    st.markdown(SUBPAGE_CSS, unsafe_allow_html=True)

# =========================================================
# 1. 页面配置
# =========================================================
if __name__ == "__main__":
    st.set_page_config(
        page_title="智能客服助手 · 小智",
        page_icon="🤖",
        layout="wide",
    )

    # =========================================================
    # 1.5 全局视觉样式（现代 AI 产品审美：浅灰背景 + 圆角气泡/按钮/折叠面板）
    # =========================================================
    st.markdown(
        """
        <style>
        /* 隐藏 Streamlit 默认菜单、页脚、顶栏 */
        #MainMenu { visibility: hidden; }
        footer { visibility: hidden; }
        header { visibility: hidden; }

        /* 主区域浅灰背景 */
        .stApp { background-color: #F8F9FA; }

        /* 聊天气泡：白底圆角 + 极轻阴影 */
        [data-testid="stChatMessage"] {
            background-color: #FFFFFF;
            border-radius: 12px;
            border: 1px solid #EEF0F2;
            box-shadow: 0 1px 3px rgba(16, 24, 40, 0.06);
        }

        /* 按钮、折叠面板统一 8px 圆角 */
        .stButton > button { border-radius: 8px; }
        [data-testid="stExpander"], details[data-testid="stExpander"] {
            border-radius: 8px;
            overflow: hidden;
        }

        /* 反馈按钮行 */
        .feedback-row .stButton > button {
            border-radius: 20px;
            padding: 2px 12px;
            font-size: 15px;
            border: 1px solid #e2e8f0;
            background: #fff;
            transition: all 0.15s;
        }
        .feedback-row .stButton > button:hover {
            transform: scale(1.08);
        }

        /* 侧边栏折叠面板间距压缩 */
        [data-testid="stSidebar"] [data-testid="stExpander"] {
            margin-top: -8px;
            margin-bottom: -8px;
        }

        /* 侧边栏内部元素间距收紧 */
        [data-testid="stSidebar"] .stVerticalBlock {
            gap: 0.3rem;
        }

        /* 反馈按钮行：列宽收缩为内容宽 + 收紧间距（三个按钮挨着排列） */
        [data-testid="stChatMessage"] [data-testid="stHorizontalBlock"] {
            gap: 0.4rem;
        }
        [data-testid="stChatMessage"] [data-testid="stColumn"] {
            flex: 0 0 auto !important;
            width: auto !important;
        }
        [data-testid="stChatMessage"] .stButton > button {
            width: auto !important;
        }

        /* 空状态欢迎卡片（居中） */
        .welcome-card { text-align: center; padding: 48px 16px 8px; }
        .welcome-card .welcome-icon { font-size: 64px; line-height: 1.2; }
        .welcome-card .welcome-title {
            font-size: 34px; font-weight: 700; margin: 12px 0 4px; color: #101828;
        }
        .welcome-card .welcome-subtitle { font-size: 16px; color: #667085; }
        </style>
        """,
        unsafe_allow_html=True,
    )

    # =========================================================
    # 3. 初始化 Session State 参数
    # =========================================================
    defaults = {
        "chunk_size": 512,
        "chunk_overlap": 100,
        "top_k": 5,
        "similarity_threshold": 0.0,
        "retrieval_mode": "向量检索",
        "kb_version": 0,  # 知识库版本号，入库/删除时 +1，用于重建 BM25 索引
        "rerank_enabled": True,  # 与置信度兜底联动：兜底依赖精排分数，必须同时开启
        "rerank_model": "bge-reranker-base（更快，CPU 友好）",
        "rerank_candidates": 10,
        "confidence_fallback_enabled": True,
        "confidence_threshold": 0.4,  # 评测期发现 Ollama 挂掉，数据不可信，先保留一次优化值
        "history_compress_enabled": True,
        "history_compress_threshold": 2000,
        "messages": [],
        "conv_id": None,  # 当前会话标识（对外 token，随机不可遍历；None 表示尚未开新会话）
        "pending_delete_hash": None,
        "pending_delete_conv": None,  # 待确认删除的会话 ID
        "evaluation_data": None,
        "evaluation_results": None,
        "evaluation_report": None,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

# =========================================================
# 3.5 访问控制说明：前端不再设密码登录（打开即用，本地自托管场景）。
#     公网防护由 API 侧承担：api.py 通过 X-API-Key 令牌鉴权（见 api.py 中间件）。
# =========================================================

# =========================================================
# 4. 运行时资源构建器（模型/向量库/链/Agent 图按「配置版本」缓存，
#    侧边栏「保存并生效」后 version +1 → 自动重建，支持本地/API 热切换）
# =========================================================
@st.cache_resource
def _build_runtime(version: int):
    """按配置版本构建全部运行时资源（进程级缓存；version 变更后重建）"""
    cfg = load_model_config()
    # 构建聊天/嵌入模型（API Key 从 .env 读取；API 配置无效时自动回退本地，保证系统可用）
    chat_model, embeddings, effective_cfg, warning = build_runtime_models(cfg)
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)
    set_vectorstore(vectorstore)  # 向检索模块注入向量库（core/retrieval.py）

    # RAG 链与意图链（全部绑定当前生效的聊天模型）
    rewrite_chain = rewrite_prompt | chat_model | StrOutputParser()
    answer_chain = answer_prompt | chat_model | StrOutputParser()
    intent_chain = intent_prompt | chat_model | StrOutputParser()
    chitchat_chain = chitchat_prompt | chat_model | StrOutputParser()
    route_llm = chat_model.bind_tools(ALL_TOOLS)

    # 意图识别 / Function Calling 路由函数（闭包绑定当前模型实例）
    classify_intent, route_intent = _make_intent_fns(intent_chain, route_llm)

    # LangGraph Agent 图
    from core.agent_graph import build_agent_graph

    agent_graph = build_agent_graph(
        {
            "route_intent": route_intent,
            "run_order_query": run_order_query,
            "run_ticket_create": run_ticket_create,
            "rewrite_chain": rewrite_chain,
            "answer_chain": answer_chain,
            "chitchat_chain": chitchat_chain,
            "do_retrieve": do_retrieve,
            "rerank_docs": rerank_docs,
            "dense_top_confidence": dense_top_confidence,
            "format_docs": format_docs,
            "intent_names": INTENT_NAMES,
            "logger": logger,
        }
    )
    return {
        "chat_model": chat_model,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "cfg": effective_cfg,  # 实际生效的配置（回退后可能与 config.yaml 不同）
        "rewrite_chain": rewrite_chain,
        "answer_chain": answer_chain,
        "intent_chain": intent_chain,
        "chitchat_chain": chitchat_chain,
        "route_llm": route_llm,
        "agent_graph": agent_graph,
        "warning": warning,
    }


def get_runtime() -> dict:
    """获取当前生效的运行时资源；配置保存后版本号递增 → 缓存自动失效重建。
    子页面（评测/知识库）也应通过本函数取资源，避免拿到切换前的旧模型。"""
    return _build_runtime(int(load_model_config().get("version", 0)))

# =========================================================
# 5. Prompt 与意图识别（链与模型实例的实际构建见 _build_runtime）
# =========================================================
INTENT_NAMES = {1: "知识库咨询", 2: "查订单", 3: "创建工单", 4: "日常闲聊"}


def _make_intent_fns(intent_chain, route_llm):
    """构建意图识别 / Function Calling 路由函数（闭包绑定当前配置的模型实例，
    保证 LangGraph 节点使用的链与当前运行时版本一致）。"""

    def classify_intent(question: str, history_text: str | None = None) -> int:
        """LLM 意图分类，解析失败时兜底为知识库咨询（1）。
        history_text 显式传入时不依赖 Streamlit 会话状态（供 LangGraph 节点调用）。"""
        if history_text is None:
            history_text = build_history_text(st.session_state.messages)
        try:
            raw = intent_chain.invoke(
                {"history": history_text, "question": question}
            ).strip()
            match = re.search(r"^\s*([1-4])\s*$", raw) or re.search(r"([1-4])", raw)
            intent = int(match.group(1)) if match else 1
        except Exception as e:
            logger.error(f"意图识别失败，兜底为知识库咨询: {e}")
            intent = 1
        logger.info(f"意图识别 | intent={intent}({INTENT_NAMES[intent]}) question={question[:50]}")
        return intent

    def route_intent(question: str, history_text: str | None = None) -> dict:
        """Function Calling 意图路由：LLM 通过 bind_tools 决定调哪个工具并提取参数。
        返回 {"intent": int, "tool_args": dict}；
        工具调用失败时降级为提示词分类（classify_intent），保证可用性。"""
        if history_text is None:
            history_text = build_history_text(st.session_state.messages)
        try:
            ai_msg = route_llm.invoke(
                route_prompt.invoke(
                    {"history": history_text, "question": question}
                ).to_messages()
            )
            tool_calls = getattr(ai_msg, "tool_calls", None) or []
            if tool_calls:
                tc = tool_calls[0]
                intent = 2 if tc["name"] == "query_order" else 3
                tool_args = dict(tc["args"] or {})
            else:
                # 未发起工具调用：按提示词约定解析 RAG / CHAT
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

    return classify_intent, route_intent


def extract_order_id(text: str) -> str | None:
    """从文本中提取订单号（6~8 位数字）—— LLM 未提取到参数时的正则兜底"""
    match = re.search(r"\d{6,8}", text)
    return match.group() if match else None


def run_order_query(question: str, tool_args: dict | None = None) -> tuple[str, str]:
    """查订单工具流程（参数来自 Function Calling 提取），返回 (回答, 工具调用说明)"""
    tool_args = tool_args or {}
    # 隐私拦截：不允许用手机号/身份证等隐私信息查询订单，引导使用订单号
    if re.search(r"1[3-9]\d{9}", question) or re.fullmatch(r"1[3-9]\d{9}", str(tool_args.get("order_id") or "")):
        return (
            "为了保护您的账户与隐私安全，暂不支持通过手机号查询订单。"
            "请在「我的订单」中找到订单号后发给我（演示环境可试用："
            f"{'、'.join(MOCK_ORDERS.keys())}），我马上为您查询。",
            "手机号查询→隐私拦截澄清",
        )
    # 参数来源优先级：LLM 提取 → 正则兜底 → 都无则主动澄清
    order_id = str(tool_args.get("order_id") or "").strip()
    if not order_id:
        order_id = extract_order_id(question) or ""
    if not order_id:
        # 主动澄清：信息不足时不瞎猜
        return (
            "好的，我来帮您查询订单。请提供您的订单号"
            f"（演示环境可试用：{ '、'.join(MOCK_ORDERS.keys()) }）。",
            "参数不足（order_id 缺失），主动澄清",
        )
    logger.info(f"调用工具 query_order | order_id={order_id}")
    order = query_order_status(order_id)
    if order is None:
        return (
            f"抱歉，没有查询到订单 {order_id} 的记录。请核对订单号是否正确，"
            "或告诉我为您登记问题、转人工处理。",
            f"query_order_status({order_id}) → 未找到",
        )
    return (
        f"已为您查到订单 {order_id} 的最新状态：\n\n"
        f"- **商品**：{order['item']}\n"
        f"- **状态**：{order['status']}\n"
        f"- **物流**：{order['carrier']}（单号 {order['tracking_no']}）\n"
        f"- **时效**：{order['eta']}\n\n"
        "还有其他需要帮忙的吗？\n\n"
        "⚠️ *演示数据：以上订单信息为功能演示，非真实订单。*",
        f"query_order_status({order_id}) → 成功",
    )


def run_ticket_create(question: str, tool_args: dict | None = None) -> tuple[str, str]:
    """创建工单工具流程（参数来自 Function Calling 提取），返回 (回答, 工具调用说明)"""
    tool_args = tool_args or {}
    issue = str(tool_args.get("issue") or "").strip() or question.strip()
    if len(issue) < 10:
        return (
            "好的，我来为您登记问题并创建工单。为了更快解决，请具体描述一下：\n\n"
            "1. 遇到了什么问题？\n2. 大概什么时间发生的？",
            "信息不足（issue 过短），主动澄清",
        )
    # 优先级：LLM 提取 → 关键词兜底（出现"投诉/紧急/马上"升级为 P1）
    priority = str(tool_args.get("priority") or "").strip().upper()
    if priority not in ("P1", "P2"):
        priority = "P1" if re.search(r"投诉|紧急|马上|立刻|严重|法院|举报", question) else "P2"
    logger.info(f"调用工具 create_ticket | priority={priority} issue={issue[:50]}")
    ticket = create_ticket_record(issue, priority=priority)
    if priority == "P1":
        prefix = "非常抱歉给您带来了不好的体验，您的情况我已加急登记：\n\n"
        suffix = "专属客服会优先处理您的工单，请您保持电话畅通，我们会第一时间与您联系。"
    else:
        prefix = "好的，您的问题我已为您登记：\n\n"
        suffix = "人工客服会按顺序跟进处理，请您留意工单进度通知。"
    return (
        prefix
        + f"- **工单号**：{ticket['ticket_id']}\n"
        f"- **优先级**：{ticket['priority']}（{ticket['sla']}）\n"
        f"- **创建时间**：{ticket['created_at']}\n\n"
        + suffix
        + "\n您可以在工单页面凭工单号查询处理进度。\n\n"
        "⚠️ *演示模式：工单未真实提交，仅供功能演示。*",
        f"create_ticket(priority={priority}) → 成功",
    )


# ---------- 闲聊链 ----------
# （chitchat_prompt 定义在此；chitchat_chain 在 _build_runtime 中按当前模型构建）
chitchat_prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            "你是智能客服助手'小智'，语气亲切、简洁。只做日常寒暄，不要编造任何公司政策或产品信息。"
            "如果用户问到业务问题，友好地引导他直接提问，你会帮他查询知识库或办理业务。",
        ),
        ("human", "{question}"),
    ]
)


# =========================================================
# 8b-2. 运行时资源装载（替代原 get_agent_graph 单图缓存）
#   模型 / 向量库 / 链 / Agent 图统一由 _build_runtime 按配置版本构建；
#   侧边栏切换模型并保存后 version 递增，此处自动拿到新资源。
#   子页面请调用 get_runtime() 获取，不要直接 import 这些模块级名字。
# =========================================================

if __name__ == "__main__":
    _RT = get_runtime()
    chat_model = _RT["chat_model"]
    embeddings = _RT["embeddings"]
    vectorstore = _RT["vectorstore"]
    set_vectorstore(vectorstore)  # 确保检索模块拿到当前向量库（防 cache 命中时未注入）
    rewrite_chain = _RT["rewrite_chain"]
    answer_chain = _RT["answer_chain"]
    agent_graph = _RT["agent_graph"]

    if _RT.get("warning"):
        st.warning(_RT["warning"])


    # =========================================================
    # 9. 页面 UI
    # =========================================================
    # 首屏：有历史消息时显示常规标题区；空状态由居中欢迎卡片替代（避免标题重复）
    if st.session_state.get("messages"):
        st.title("🤖 智能客服助手「小智」")
        st.caption("意图路由 · RAG 知识库问答 · 工具调用（查订单/建工单） · Rerank 精排 · 置信度兜底")
        st.success("系统加载成功 ✅ 试试：查订单 2024001 / 退货政策 / 我要投诉")

    # =========================================================
    # 10. 侧边栏
    # =========================================================
    with st.sidebar:
        # ---------- 新建会话（常驻顶部，左对齐） ----------
        if st.button("➕ 新建会话"):
            st.session_state.messages = []
            st.session_state.conv_id = None
            st.rerun()

        # ---------- 快捷问题（常驻侧边栏，提问后不消失） ----------
        st.markdown("**💡 快捷问题**")
        _welcome_qs = [
            "🛒 查订单 2024001",
            "❓ 帮我查下订单到哪了",
            "🎫 我要投诉，耳机质量有问题",
            "👋 你好，你都能做什么？",
        ]
        for _wq in _welcome_qs:
            if st.button(_wq, key=f"quick_{_wq}", use_container_width=True):
                st.session_state.pending_question = _wq.split(" ", 1)[-1] if _wq[0] in "🛒📚🎫👋" else _wq
                st.rerun()
        st.divider()

        # ---------- 嵌入空间守卫（仅异常时提示；模型配置入口只在「模型设置」页） ----------
        _cfg = load_model_config()
        # 当前生效嵌入空间（运行时实际使用，回退后与 config.yaml 可能不同） vs 知识库建库空间
        _cur_space = embedding_space(_RT["cfg"])
        _kb_space = _cfg.get("kb_embedding_space", LOCAL_EMBEDDING_MODEL)
        _kb_ok = _cur_space == _kb_space

        # ---------- 嵌入空间不一致：阻断提示 + 跳转模型设置页（重建入口已收敛至该页） ----------
        if not _kb_ok:
            st.error(
                f"⚠️ 检索已暂停：当前嵌入向量空间（{_cur_space}）与知识库建库空间"
                f"（{_kb_space}）不一致，混用会导致检索结果错乱。请到「🧠 模型设置」页完成迁移。"
            )
            st.page_link("pages/04_模型设置.py", label="🔄 前往模型设置页重建知识库")

        # ---------- 会话记录（轻量直出：无边框灰度列表，导出按钮收纳在下方） ----------
        conversations = db_list_conversations()
        if conversations:
            st.markdown(f"**💬 会话记录（{len(conversations)}）**")
            with st.container(height=260):
                if not conversations:
                    st.caption("暂无历史会话，提问后将自动保存。")
                for conv_id, title, created_at in conversations:
                    is_current = st.session_state.get("conv_id") == conv_id
                    label = f"{'🟢' if is_current else '💬'} {title}（{created_at[5:16]}）"

                    if st.session_state.get("pending_delete_conv") == conv_id:
                        st.warning(f"确认删除会话「{title}」？该操作不可恢复。")
                        col_ok, col_cancel = st.columns(2)
                        with col_ok:
                            if st.button("✓ 确认删除", key=f"conv_del_ok_{conv_id}", type="primary"):
                                db_delete_conversation(conv_id)
                                logger.info(f"会话删除 | conv_id={conv_id} title={title[:20]}")
                                if is_current:
                                    st.session_state.messages = []
                                    st.session_state.conv_id = None
                                st.session_state.pending_delete_conv = None
                                st.rerun()
                        with col_cancel:
                            if st.button("✗ 取消", key=f"conv_del_cancel_{conv_id}"):
                                st.session_state.pending_delete_conv = None
                                st.rerun()
                    else:
                        col_open, col_del = st.columns([0.85, 0.15])
                        with col_open:
                            if st.button(label, key=f"conv_{conv_id}", use_container_width=True):
                                if not is_current:
                                    st.session_state.conv_id = conv_id
                                    st.session_state.messages = db_load_messages(conv_id)
                                    st.rerun()
                        with col_del:
                            if st.button("🗑️", key=f"conv_del_{conv_id}", help="删除此会话"):
                                st.session_state.pending_delete_conv = conv_id
                                st.rerun()

            export_df = pd.DataFrame(
                db_export_all_messages(),
                columns=["会话ID", "会话标题", "会话创建时间", "角色", "消息内容",
                         "意图", "路由说明", "反馈", "消息时间"],
            )
            st.download_button(
                "📥 导出全部会话 (CSV)",
                data=export_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"all_conversations_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
                use_container_width=True,
            )

            if st.session_state.get("conv_id"):
                current_title = next(
                    (t for cid, t, _ in conversations if cid == st.session_state.conv_id),
                    f"会话{st.session_state.conv_id}",
                )
                md_lines = [f"# 会话记录：{current_title}", ""]
                for msg in st.session_state.messages:
                    role_label = "👤 用户" if msg["role"] == "user" else "🤖 助手"
                    md_lines += [f"**{role_label}**", "", msg.get("content", ""), ""]
                st.download_button(
                    "📥 导出当前会话 (Markdown)",
                    data="\n".join(md_lines).encode("utf-8"),
                    file_name=f"conversation_{st.session_state.conv_id}.md",
                    mime="text/markdown",
                    use_container_width=True,
                )

        # ---------- （原「管理后台入口」已移除：日志功能迁入「模型设置」页底部折叠面板，
        #            不再保留独立入口，避免侧边栏多一个几乎用不到的页面） ----------

    # =========================================================
    # 11. 聊天界面
    # =========================================================


    @st.cache_resource
    def _get_ocr_engine():
        """RapidOCR 引擎单例（与知识库入库同款本地引擎，初始化约 1.7s 缓存复用）"""
        from rapidocr import RapidOCR

        return RapidOCR()


    def _ocr_image_to_text(image_file) -> str:
        """聊天图片 OCR（RapidOCR 本地推理，中英文；异常返回空串，由调用方提示）。
        注：本机 D:\\tesseract\\tesseract-ocr.exe 损坏（WinError 740 需提权），
        故弃用 pytesseract 改用 rapidocr（Docling 同款，纯 onnxruntime 无系统依赖）"""
        try:
            img = Image.open(image_file).convert("RGB")
            result = _get_ocr_engine()(np.array(img))
            txts = getattr(result, "txts", None) or ()
            return "\n".join(txts).strip()
        except Exception as e:
            logger.warning(f"图片 OCR 失败 | error={e}")
            return ""


    def render_reference_doc(i: int, doc):
        """引用文献紧凑渲染：来源一行 + 相关性彩色 Badge（三档色阶替代 progress 条）"""
        metadata = doc.metadata or {}
        source = metadata.get("source", "未知来源")
        chunk_id = metadata.get("chunk_id", "未知文档块")
        st.markdown(f"**📄 {source}** (Chunk: {chunk_id})")
        if "rerank_score" in metadata:
            score = float(metadata["rerank_score"])
            if score >= 0.7:
                bg, fg = "#f6ffed", "#52c41a"  # 绿：高相关
            elif score >= 0.4:
                bg, fg = "#e6f7ff", "#1890ff"  # 蓝：中相关
            else:
                bg, fg = "#fff7e6", "#fa8c16"  # 橙：低相关
            st.markdown(
                f'<span style="background-color:{bg}; color:{fg}; padding:2px 8px; '
                f'border-radius:10px; font-size:12px;">相关性: {score:.2f}</span>',
                unsafe_allow_html=True,
            )
        st.write(doc.page_content)


    def render_feedback(message: dict, idx: int, show_regen: bool = False):
        """回答下方的点赞/点踩/重新生成操作行"""
        fb = message.get("feedback")
        _up_label = "👍 有帮助" if fb == "up" else "👍"
        _down_label = "👎 待改进" if fb == "down" else "👎"

        st.markdown('<div class="feedback-row">', unsafe_allow_html=True)
        # 窄比例列：按钮靠左聚拢（等分列会把 👍👎🔄 拉到整行宽度，间距过宽）
        if show_regen:
            _c1, _c2, _c3 = st.columns([0.07, 0.07, 0.14])
        else:
            _c1, _c2 = st.columns([0.07, 0.07])
        with _c1:
            if st.button(_up_label, key=f"fb_up_{idx}",
                         type="primary" if fb == "up" else "secondary"):
                st.session_state.messages[idx]["feedback"] = "up"
                if message.get("msg_db_id"):
                    db_update_feedback(message["msg_db_id"], "up")
                logger.info(f"用户反馈 | score=up")
                st.rerun()
        with _c2:
            if st.button(_down_label, key=f"fb_down_{idx}",
                         type="primary" if fb == "down" else "secondary"):
                st.session_state.messages[idx]["feedback"] = "down"
                if message.get("msg_db_id"):
                    db_update_feedback(message["msg_db_id"], "down")
                logger.info(f"用户反馈 | score=down")
                st.rerun()
        if show_regen:
            with _c3:
                if st.button("🔄 重新生成", key=f"regen_{idx}", help="删除这条回答并重新生成"):
                    if message.get("msg_db_id"):
                        db_delete_message(message["msg_db_id"])
                    st.session_state.messages.pop(idx)
                    st.session_state.pending_question = st.session_state.messages[-1]["content"]
                    st.session_state.regenerate = True
                    st.rerun()
        st.markdown('</div>', unsafe_allow_html=True)


    # ---------- 空状态：居中欢迎卡片 ----------
    if not st.session_state.messages:
        st.markdown(
            """
            <div class="welcome-card">
                <div class="welcome-icon">🤖</div>
                <div class="welcome-title">小智</div>
                <div class="welcome-subtitle">您的智能客服助手，支持查订单、建工单、知识库问答</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    else:
        for msg_idx, message in enumerate(st.session_state.messages):
            with st.chat_message(message["role"]):
                if message["role"] == "assistant" and "intent_name" in message:
                    st.caption(f"🧭 意图识别：{message['intent_name']} → {message.get('route_note', '')}")
                st.markdown(message["content"])
                if message["role"] == "assistant":
                    if message.get("rewritten_question"):
                        with st.expander("查看改写后的检索问题"):
                            st.write(message["rewritten_question"])
                    if "docs" in message and message["docs"]:
                        mode = message.get("retrieval_mode", "")
                        mode_label = f"（{mode}）" if mode else ""
                        with st.expander(f"查看检索到的参考资料{mode_label}"):
                            for i, doc in enumerate(message["docs"], start=1):
                                render_reference_doc(i, doc)
                    # 轻量指标行（新生成的回答用 metric 卡，历史消息压缩为一行）
                    if message.get("total_time") is not None:
                        st.caption(
                            f"⏱️ 总耗时 {message['total_time']:.1f}s | "
                            f"🔍 检索命中 {message.get('doc_count', 0)} 条 | "
                            f"🎯 意图：{message.get('intent_name', '未知')} | "
                            f"📮 {message.get('tokens', 0):,} tokens"
                        )
                    render_feedback(
                        message, msg_idx,
                        show_regen=(msg_idx == len(st.session_state.messages) - 1),
                    )

    # =========================================================
    # 12. 用户输入与回答（Agent 路由入口，流式输出）
    # =========================================================
    # Streamlit 1.63：accept_file=True 时提交后返回 ChatInputValue（.text/.files），
    # 未提交返回 None（不能解包）；accept_file=False 时才是纯 str
    _prompt = st.chat_input(
        "请输入你的问题，支持：知识库咨询 / 查订单 / 登记工单...",
        accept_file=True,
        file_type=["png", "jpg", "jpeg", "bmp", "tiff"],
    )
    if _prompt and not isinstance(_prompt, str):
        user_question = (_prompt.text or "").strip()
        uploaded_image = _prompt.files[0] if _prompt.files else None
    else:
        user_question, uploaded_image = _prompt, None

    # 演示问题按钮点击后，将预置问题作为用户输入
    if not user_question and st.session_state.get("pending_question"):
        user_question = st.session_state.pop("pending_question")

    # 多模态输入：图片 OCR 结果并入提问（仅图→OCR 即问题；文字+图→拼接）
    if uploaded_image is not None:
        with st.spinner("🔍 正在识别图片文字（OCR）..."):
            _ocr_text = _ocr_image_to_text(uploaded_image)
        if not _ocr_text:
            st.warning("⚠️ 图片中未识别到文字，请重新上传或直接输入问题。")
            st.stop()
        user_question = (
            f"{user_question}\n\n【图片识别内容】\n{_ocr_text}"
            if user_question
            else _ocr_text
        )

    if user_question:
        # 嵌入空间守卫：当前嵌入模型与知识库建库空间不一致时阻断问答，防止检索错乱
        if not _kb_ok:
            st.error(
                f"嵌入模型向量空间（{_cur_space}）与知识库建库空间（{_kb_space}）不一致，"
                "检索已暂停。请到左侧导航「🧠 模型设置」页完成迁移后再提问。"
            )
            st.stop()

        # 首次提问时创建会话记录（标题取问题前 20 字）
        if st.session_state.get("conv_id") is None:
            st.session_state.conv_id = db_create_conversation(user_question[:20])

        # 「重新生成」：user 消息已在前端历史与 DB 中，跳过重复落库/追加/渲染
        _is_regen = st.session_state.pop("regenerate", False)

        if not _is_regen:
            db_save_message(
                st.session_state.conv_id, {"role": "user", "content": user_question}
            )

            with st.chat_message("user"):
                st.markdown(user_question)
                if uploaded_image is not None:
                    st.image(uploaded_image, width=220)
            st.session_state.messages.append({"role": "user", "content": user_question})

        with st.chat_message("assistant"):
            # ---- LangGraph 流式执行 ----
            # messages 模式：天然透传节点内 LLM 的逐 token 输出（生成节点）
            # values 模式：每个节点执行完返回完整状态（用于意图徽章/轨迹/资料）
            from core.agent_graph import ANSWER_NODES

            state_holder: dict = {}
            badge_slot = st.empty()
            badge_slot.caption("🧭 正在识别意图并检索资料...")

            # 思考过程可视化：运行时展开逐节点状态与耗时，完成后自动折叠（可点开回看）
            _t0 = time.time()
            status_box = st.status("🤖 小智正在思考...", expanded=True)
            with status_box:
                node_slot = st.empty()
            _node_lines: list = []
            _clock = {"last": _t0}
            _collector = _NodeTokenCollector()

            def _graph_stream():
                # 历史文本：可选压缩（侧边栏「多轮上下文压缩」开关）
                _chat_model = _RT["chat_model"]
                _summary_prompt = (
                    "你是一名对话摘要助手。请把下面这段早期客服对话压缩成 3-5 条要点，"
                    "保留关键事实（订单号/产品/问题结论等），丢掉寒暄和重复信息。"
                    "只输出要点列表，每条一行，不要解释。\n\n"
                    "{history}"
                )

                def _summarizer(text: str) -> str:
                    try:
                        from langchain_core.prompts import PromptTemplate
                        from langchain_core.output_parsers import StrOutputParser
                        chain = PromptTemplate.from_template(_summary_prompt) | _chat_model | StrOutputParser()
                        return chain.invoke({"history": text})
                    except Exception as e:
                        logger.warning(f"历史压缩失败，降级不压缩: {e}")
                        raise

                history_text = build_history_text(
                    st.session_state.messages,
                    max_tokens=1500,
                    compress=st.session_state.get("history_compress_enabled", True),
                    compress_threshold_tokens=st.session_state.get("history_compress_threshold", 2000),
                    summarizer=_summarizer if st.session_state.get("history_compress_enabled", True) else None,
                )
                # 显式同步前端选择的 Rerank 模型，保证图节点使用的精排模型与设置页一致
                set_rerank_model(st.session_state.get("rerank_model"))
                initial_state = {
                    "question": user_question,
                    "history": history_text,
                    "top_k": st.session_state.top_k,
                    "similarity_threshold": st.session_state.similarity_threshold,
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "rerank_enabled": st.session_state.get("rerank_enabled", False),
                    "rerank_candidates": st.session_state.get("rerank_candidates", 10),
                    "confidence_fallback_enabled": st.session_state.get(
                        "confidence_fallback_enabled", True
                    ),
                    "confidence_threshold": st.session_state.get("confidence_threshold", 0.5),
                    "active_sources": get_active_sources(),
                }
                stream = agent_graph.stream(
                    initial_state,
                    config={"callbacks": [_collector]},
                    stream_mode=["messages", "values"],
                )
                for mode, chunk in stream:
                    if mode == "values":
                        state_holder["state"] = chunk
                        if chunk.get("intent_name"):
                            # 意图节点完成后立即展示路由徽章
                            badge_slot.caption(
                                f"🧭 意图识别：{chunk['intent_name']} → {chunk.get('route_note', '')}"
                            )
                        # 逐节点耗时：values 事件里 trace 的最后一项是刚完成的节点
                        _now = time.time()
                        _elapsed = _now - _clock["last"]
                        _clock["last"] = _now
                        _trace = chunk.get("trace") or []
                        if _trace:
                            _node = _trace[-1]
                            _detail = ""
                            if _node == "意图识别(Function Calling)" and chunk.get("intent_name"):
                                _detail = f" → {chunk['intent_name']}"
                            elif _node in ("知识检索", "Rerank精排"):
                                _detail = f"，命中 {len(chunk.get('docs') or [])} 条"
                            _node_lines.append(f"✅ **{_node}**（{_elapsed:.1f}s）{_detail}")
                            node_slot.markdown("\n\n".join(_node_lines))
                    elif mode == "messages":
                        _msg, meta = chunk
                        if meta.get("langgraph_node") in ANSWER_NODES:
                            content = getattr(_msg, "content", "")
                            if isinstance(content, str) and content:
                                yield content

            # 阶段一/二合并：图执行 + 生成节点逐字流式输出
            try:
                full_answer = st.write_stream(_graph_stream())
            except Exception as e:
                # 生成失败兜底：友好提示 + 错误消息正常落库，避免"只有问没有答"的断头会话
                _total_time = time.time() - _t0
                status_box.update(label="❌ 生成失败", state="error", expanded=False)
                logger.error(
                    f"回答生成失败 question={user_question[:50]}: {e}", exc_info=True
                )
                full_answer = (
                    "⚠️ 回答生成失败，请点击下方「🔄 重新生成」重试；"
                    "若持续失败请到「🧠 模型设置」检查模型服务状态。"
                )
                st.markdown(full_answer)
                _fail_state = state_holder.get("state", {})
                _fail_msg = {
                    "role": "assistant",
                    "content": full_answer,
                    "intent_name": _fail_state.get("intent_name", "未知"),
                    "route_note": "生成失败降级",
                    "rewritten_question": None,
                    "docs": [],
                    "retrieval_mode": st.session_state.retrieval_mode,
                    "total_time": round(_total_time, 2),
                    "doc_count": 0,
                    "tokens": 0,
                }
                st.session_state.messages.append(_fail_msg)
                _fail_msg["msg_db_id"] = db_save_message(
                    st.session_state.conv_id, _fail_msg
                )
                render_feedback(
                    st.session_state.messages[-1],
                    len(st.session_state.messages) - 1,
                    show_regen=True,
                )
                st.stop()

            _total_time = time.time() - _t0
            status_box.update(label="✅ 思考完成", state="complete", expanded=False)

            final_state = state_holder.get("state", {})
            intent_name = final_state.get("intent_name", "未知")
            route_note = final_state.get("route_note", "")
            rewritten_q = final_state.get("rewritten_q")
            docs = final_state.get("docs", [])

            # 工具调用 / 兜底分支没有 LLM token，直接渲染最终文本
            if not full_answer and final_state.get("answer"):
                full_answer = final_state["answer"]
                st.markdown(full_answer)

            # token 用量：业务链路真实采集（本地 Ollama 可靠；个别 API 流式不返回 usage 时为 0）
            _usage = _collector.usage_by_node or {}
            _tokens = sum(u.get("total_tokens", 0) for u in _usage.values())

            # 运行指标：一行轻量展示（与历史消息格式一致，不占空间）
            st.caption(
                f"⏱️ 总耗时 {_total_time:.1f}s | 🔍 检索命中 {len(docs)} 条 | "
                f"🎯 意图：{intent_name} | 📮 {_tokens:,} tokens"
            )
            if rewritten_q:
                with st.expander("查看改写后的检索问题"):
                    st.write(rewritten_q)
            with st.expander(
                f"查看检索到的参考资料（{st.session_state.retrieval_mode}，共 {len(docs)} 条）"
            ):
                if docs:
                    for i, doc in enumerate(docs, start=1):
                        render_reference_doc(i, doc)
                else:
                    st.write("本次回答未使用知识库检索。")

            logger.info(
                f"问答完成 | Graph 路由={intent_name} mode={st.session_state.retrieval_mode} "
                f"rerank={st.session_state.get('rerank_enabled', False)} "
                f"top_k={st.session_state.top_k} 命中={len(docs)} question={user_question[:50]}"
            )

            # 阶段三：消息落库 + 反馈按钮（先入历史再渲染，保证刷新后仍在）
            assistant_msg = {
                "role": "assistant",
                "content": full_answer,
                "intent_name": intent_name,
                "route_note": route_note,
                "rewritten_question": rewritten_q,
                "docs": docs,
                "retrieval_mode": st.session_state.retrieval_mode,
                "total_time": round(_total_time, 2),
                "doc_count": len(docs),
                "tokens": _tokens,
            }
            st.session_state.messages.append(assistant_msg)
            assistant_msg["msg_db_id"] = db_save_message(
                st.session_state.conv_id, assistant_msg
            )
            render_feedback(
                st.session_state.messages[-1],
                len(st.session_state.messages) - 1,
                show_regen=True,
            )