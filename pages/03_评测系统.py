# =========================================================
# 评测系统（统一参数 + 评测类型切换 + 标签页结果）
# =========================================================

import io
import hashlib
import json
import logging
import os
import re
import shutil
import time
import uuid
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

from 智能客服助手 import (
    get_runtime, _make_intent_fns, run_order_query, run_ticket_create,
    INTENT_NAMES, chitchat_prompt, logger,
)
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser
from core.chain_eval import (
    NODE_LABELS, STATUS_ICON, compute_semantic_similarity,
    normalize_intent, run_chain_eval_single,
)
from core.eval_langsmith import (
    build_model_info, config_fingerprint,
    is_langsmith_ready, set_api_key, upload_chain_experiment,
)
from core.model_factory import (
    kb_compatibility, load_model_config,
    CHAT_PRESETS, EMBEDDING_PRESETS,
    describe_source,
    build_chat_model, build_embeddings, embedding_space,
)
from core.prompts import rewrite_prompt, answer_prompt, intent_prompt, route_prompt
from core.tools import ALL_TOOLS
from core.agent_graph import build_agent_graph
from core.retrieval import (
    do_retrieve, format_docs, rerank_docs, SPLIT_STRATEGIES, rebuild_knowledge_base,
    build_bm25_index_for_vectorstore,
)

st.set_page_config(page_title="评测系统", page_icon="🧪", layout="wide")

# =========================================================
# 轻量页面样式（与知识库管理页一致：白底、细边框、弱化装饰）
# =========================================================
st.markdown(
    """
    <style>
      /* 与主页/其他子页面一致：隐藏默认菜单、页脚、Deploy 入口，统一背景色 */
      #MainMenu { visibility: hidden; }
      footer { visibility: hidden; }
      header { visibility: hidden; }
      .stApp { background-color: #F8F9FA; }
      .metric-card {
        background: #fff; border: 1px solid #e2e8f0; border-radius: 8px;
        padding: 12px 14px; margin-bottom: 6px;
      }
      .metric-card .m-label { font-size: 12px; color: #64748b; margin-bottom: 2px; font-weight: 500; }
      .metric-card .m-value { font-size: 20px; font-weight: 700; color: #1e293b; line-height: 1.2; }
      .metric-card .m-help { font-size: 11px; color: #94a3b8; margin-top: 3px; }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("🧪 评测系统")
st.caption("回复质量评测看「答得好不好」——针对回复结果的测量（忠实度、检索相关性）；全链路评测看「过程对不对」——针对执行链路的测量（意图路由、节点耗时、Token 消耗）。")


def _metric_cards(items, cols=4):
    """渲染一组指标卡。items: [(标签, 数值, 说明), ...]"""
    columns = st.columns(cols)
    for i, item in enumerate(items):
        label, value = item[0], item[1]
        help_text = item[2] if len(item) > 2 else ""
        with columns[i % cols]:
            st.markdown(
                f"""
                <div class="metric-card">
                  <div class="m-label">{label}</div>
                  <div class="m-value">{value}</div>
                  <div class="m-help">{help_text or ''}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def _section(title):
    st.subheader(title, divider="gray")


MODES = ["向量检索", "关键词", "混合检索"]

# 节点名 → 短名（Token 明细行展示用）
NODE_SHORT = {
    "intent": "路由", "rag_retrieve": "改写+检索", "rag_answer": "回复",
    "order": "查订单", "ticket": "建工单", "chitchat": "闲聊",
}

# 运行时资源：每次脚本执行都取当前生效版本（主页切换模型后自动跟随）
_RT = get_runtime()
chat_model = _RT["chat_model"]
embeddings = _RT["embeddings"]
rewrite_chain = _RT["rewrite_chain"]
answer_chain = _RT["answer_chain"]
agent_graph = _RT["agent_graph"]

# 全链路评测依赖（真实生产图 + 裁判 LLM + 向量模型），core/chain_eval.py 框架无关
CHAIN_EVAL_DEPS = {
    "graph": agent_graph,
    "chat_model": chat_model,
    "embeddings": embeddings,
}

# 嵌入空间守卫：与知识库建库空间不一致时评测结果不可信（用运行时实际生效配置判断）
_KB_OK, _CUR_SPACE, _KB_SPACE = kb_compatibility(_RT["cfg"])
if not _KB_OK:
    st.error(
        f"⚠️ 检索已暂停：当前嵌入向量空间（{_CUR_SPACE}）与知识库建库空间（{_KB_SPACE}）"
        "不一致。请先到左侧导航「🧠 模型设置」页完成迁移，否则检索类评测结果不可信。"
    )
    st.page_link("pages/04_模型设置.py", label="🔄 前往模型设置页重建知识库")

# =========================================================
# 评测模型能力：从模型设置页的模型库点选，构建独立评测运行时
# =========================================================
_EVAL_KB_ROOT = Path(__file__).resolve().parent.parent / "chroma_eval"


def _evaluation_cfg() -> dict:
    """评测配置：完全跟随全局线上配置（聊天模型来自「模型设置」页；嵌入固定与主页库一致）。"""
    cfg = load_model_config()
    cfg["embedding"] = dict(_RT["cfg"].get("embedding", {}))
    return cfg


def _evaluation_context_key(cfg: dict, strategy: str, chunk_size: int) -> str:
    payload = json.dumps(
        {"chat": cfg.get("chat"), "embedding": cfg.get("embedding"), "strategy": strategy, "chunk_size": chunk_size},
        ensure_ascii=False, sort_keys=True,
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:16]


@st.cache_resource
def _build_evaluation_context(context_key: str, cfg_json: str, strategy: str, chunk_size: int) -> dict:
    """构建评测运行时：直接复用主页 chroma_db（不再重建独立评测库）。
    之前是 `chroma_eval/<fingerprint>/` 新建空库后 rebuild → 依赖 Ollama 嵌入；
    Ollama 挂时 rebuild 失败 → 评测库空 → 任何检索都返回 0 文档。
    现在改为直接用主页库，与生产环境 100% 一致；embedding 函数仍按 cfg 创建（仅供可能用到的 API 嵌入路径）。"""
    cfg = json.loads(cfg_json)
    chat = build_chat_model(cfg)
    embeddings = build_embeddings(cfg)
    kb_dir = Path(__file__).resolve().parent.parent / "chroma_db"  # 主页库，不新建独立库
    vectorstore = Chroma(persist_directory=str(kb_dir), embedding_function=embeddings)
    bm25_data = build_bm25_index_for_vectorstore(vectorstore)

    rewrite = rewrite_prompt | chat | StrOutputParser()
    answer = answer_prompt | chat | StrOutputParser()
    intent = intent_prompt | chat | StrOutputParser()
    route_llm = chat.bind_tools(ALL_TOOLS)
    _classify, route = _make_intent_fns(intent, route_llm)

    def _eval_retrieve(query, top_k, similarity_threshold, retrieval_mode, sources=None):
        return do_retrieve(
            query, top_k, similarity_threshold, retrieval_mode, sources,
            vs=vectorstore, bm25_data=bm25_data,
        )

    graph = build_agent_graph({
        "route_intent": route,
        "run_order_query": run_order_query,
        "run_ticket_create": run_ticket_create,
        "rewrite_chain": rewrite,
        "answer_chain": answer,
        "chitchat_chain": chitchat_prompt | chat | StrOutputParser(),
        "do_retrieve": _eval_retrieve,
        "rerank_docs": rerank_docs,
        "format_docs": format_docs,
        "intent_names": INTENT_NAMES,
        "logger": logger,
    })
    return {
        "chat_model": chat,
        "embeddings": embeddings,
        "vectorstore": vectorstore,
        "rewrite_chain": rewrite,
        "answer_chain": answer,
        "graph": graph,
        "do_retrieve": _eval_retrieve,
        "cfg": cfg,
        "kb_dir": str(kb_dir),
        "deps": {
            "graph": graph,
            "chat_model": chat,
            "embeddings": embeddings,
        },
    }


def _read_csv_auto(csv_file) -> pd.DataFrame:
    """自动识别编码读 CSV：覆盖 UTF-8 (含 BOM) / UTF-8 / GBK / GB18030 / Latin-1。
    Excel 中文版默认存 GBK，直接 pd.read_csv 默认 UTF-8 会解析失败。"""
    raw = csv_file.getvalue()
    last_err = None
    for enc in ["utf-8-sig", "utf-8", "gbk", "gb18030", "latin-1"]:
        try:
            return pd.read_csv(io.StringIO(raw.decode(enc)))
        except UnicodeDecodeError as e:
            last_err = e
            continue
    raise last_err or UnicodeDecodeError("无法识别 CSV 编码（已尝试 utf-8-sig/utf-8/gbk/gb18030/latin-1）")


def _samples_from_dataframe(df: pd.DataFrame) -> list[dict]:
    """把 CSV 标准化为可持久化的评测样本。"""
    if "question" not in df.columns:
        raise ValueError("CSV 缺少 'question' 列")
    samples = []
    for _, row in df.iterrows():
        if pd.isna(row["question"]):
            continue
        question = str(row["question"]).strip()
        if not question:
            continue
        samples.append({
            "question": question,
            "ground_truth": (
                str(row["ground_truth"]).strip()
                if "ground_truth" in df.columns and pd.notna(row.get("ground_truth")) else ""
            ),
            "expected_intent": (
                str(row["expected_intent"]).strip()
                if "expected_intent" in df.columns and pd.notna(row.get("expected_intent")) else ""
            ),
        })
    return samples


def collect_questions():
    """读取已确认导入的评测样本，返回问题、标准答案、期望意图。"""
    samples = st.session_state.get("eval_imported_samples") or []
    if not samples:
        st.warning("请先在下方「导入评测数据」中导入 CSV 或手动填写问题。")
        return None
    questions = [s["question"] for s in samples]
    ground_truths = {s["question"]: s["ground_truth"] for s in samples if s.get("ground_truth")}
    expected_intents = {s["question"]: s["expected_intent"] for s in samples if s.get("expected_intent")}
    return questions, ground_truths, expected_intents


@st.dialog("导入评测数据")
def _import_eval_data_dialog():
    """弹窗导入：用户确认后才写入评测样本。"""
    csv_tab, manual_tab = st.tabs(["CSV 文件导入", "手动输入"])
    with csv_tab:
        st.caption("① 下载模板并按格式填写  ② 上传 CSV 后确认导入")
        template_df = pd.DataFrame({
            "question": ["退货政策是怎么规定的？", "查订单 2024001", "我要投诉，耳机有严重质量问题必须马上处理", "你好，你都能做什么？"],
            "expected_intent": ["知识库咨询", "查订单", "创建工单", "日常闲聊"],
            "ground_truth": ["7天无理由退货，质量问题15天内可退...", "订单状态：已发货...", "工单号：Txxx，优先级 P1...", "我可以帮您查知识库、查订单、建工单..."],
        })
        st.download_button("下载 CSV 模板", template_df.to_csv(index=False).encode("utf-8-sig"),
                           file_name="评测模板.csv", mime="text/csv", use_container_width=True)
        csv_file = st.file_uploader("上传 CSV（必须含 question；可选 expected_intent、ground_truth）", type=["csv"], key="eval_import_csv")
        if csv_file is not None:
            try:
                preview_df = _read_csv_auto(csv_file)
                samples = _samples_from_dataframe(preview_df)
                st.success(f"已识别：{csv_file.name} ｜ {len(samples)} 条问题 ｜ 标准答案：{'有' if 'ground_truth' in preview_df.columns else '无'} ｜ 期望意图：{'有' if 'expected_intent' in preview_df.columns else '无'}")
                if st.button("确认导入 CSV", type="primary", use_container_width=True, key="confirm_import_csv"):
                    if not samples:
                        st.warning("文件中没有有效问题。")
                    else:
                        st.session_state["eval_imported_samples"] = samples
                        st.session_state["eval_imported_meta"] = {
                            "name": csv_file.name, "source": "CSV 文件", "count": len(samples),
                            "has_ground_truth": "ground_truth" in preview_df.columns,
                            "has_expected_intent": "expected_intent" in preview_df.columns,
                        }
                        st.rerun()
            except Exception as e:
                st.error(f"文件解析失败：{e}")
    with manual_tab:
        manual_text = st.text_area("每行一个问题", placeholder="例如：\n退货政策是怎么规定的？\n耳机保修多久？", height=180, key="eval_import_manual")
        if st.button("确认导入手动问题", type="primary", use_container_width=True, key="confirm_import_manual"):
            samples = [{"question": q.strip(), "ground_truth": "", "expected_intent": ""} for q in manual_text.split("\n") if q.strip()]
            if not samples:
                st.warning("请至少填写一个问题。")
            else:
                st.session_state["eval_imported_samples"] = samples
                st.session_state["eval_imported_meta"] = {"name": "手动输入问题", "source": "手动输入", "count": len(samples), "has_ground_truth": False, "has_expected_intent": False}
                st.rerun()


def _render_eval_data_card():
    """数据入口：在 Step ② 容器内渲染，不再自带边框和标题。"""
    meta = st.session_state.get("eval_imported_meta") or {}
    samples = st.session_state.get("eval_imported_samples") or []
    if samples:
        extras = []
        if meta.get("has_ground_truth"):
            extras.append("含标准答案")
        if meta.get("has_expected_intent"):
            extras.append("含期望意图")
        st.markdown(f"**✅ {meta.get('name', '评测数据')}** 　`{meta.get('count', len(samples))} 条`" + (" 　".join(f"`{e}`" for e in extras) if extras else ""))
    else:
        st.markdown("**未导入** 　请点击下方按钮导入 CSV 或手动输入")
    _b1, _b2 = st.columns([1, 4])
    with _b1:
        if st.button("查看 / 更换" if samples else "导入数据", use_container_width=True, key="open_eval_import"):
            _import_eval_data_dialog()
    if samples and st.button("清除当前数据", key="clear_eval_import"):
        st.session_state.pop("eval_imported_samples", None)
        st.session_state.pop("eval_imported_meta", None)
        st.rerun()


def run_eval_single(question: str, ground_truth: str, mode: str,
                    rerank_on: bool, rerank_candidate_k: int,
                    top_k: int, fallback_on: bool, fallback_th: float,
                    runtime: dict | None = None,
                    similarity_threshold: float = 0.0):
    runtime = runtime or {}
    _rewrite_chain = runtime.get("rewrite_chain", rewrite_chain)
    _answer_chain = runtime.get("answer_chain", answer_chain)
    _judge_model = runtime.get("chat_model", chat_model)
    _eval_embeddings = runtime.get("embeddings", embeddings)
    _retrieve = runtime.get("do_retrieve", do_retrieve)
    start_time = time.time()
    result = {
        "question": question, "ground_truth": ground_truth, "mode": mode,
        "rerank": rerank_on, "fallback": False, "rewritten": "", "answer": "",
        "docs": [], "retrieval_time": 0.0, "generation_time": 0.0, "total_time": 0.0,
        "doc_count": 0, "faithfulness": 0.0, "relevance": 0.0,
        "retrieval_relevance": 0.0, "semantic_similarity": 0.0, "error": None,
    }
    try:
        result["rewritten"] = _rewrite_chain.invoke({"history": "", "question": question}).strip()
    except Exception as e:
        result["error"] = f"改写失败: {e}"
        result["rewritten"] = question
    rerank_degraded = False
    try:
        docs = _retrieve(result["rewritten"], top_k=rerank_candidate_k if rerank_on else top_k,
                         similarity_threshold=similarity_threshold, retrieval_mode=mode)
        if rerank_on and docs:
            docs = rerank_docs(result["rewritten"], docs, top_k)
            # 精排服务失败时返回原始排序并打降级标记：没有真实分数，不能当满分
            rerank_degraded = bool(docs[0].metadata.get("_rerank_degraded"))
        result["docs"] = docs
        result["doc_count"] = len(docs)
    except Exception as e:
        result["error"] = (result["error"] or "") + f" | 检索失败: {e}"
        docs = []
    result["retrieval_time"] = round(time.time() - start_time, 3)
    fallback_triggered = False
    if rerank_on and docs and fallback_on:
        # 降级时无真实 rerank 分，按 0 分处理：强制进入兜底判定，避免垃圾资料被当高相关放行
        top_score = 0.0 if rerank_degraded else float(docs[0].metadata.get("rerank_score", 1.0))
        if rerank_degraded:
            result["error"] = ((result["error"] + " | ") if result["error"] else "") + \
                "精排服务不可用已降级原始排序（rerank 分数不可信，按未精排判定）"
        if top_score < fallback_th:
            fallback_triggered = True
            result["fallback"] = True
            result["answer"] = "【置信度兜底：未找到高度相关内容】"
            result["generation_time"] = 0.0
    if not fallback_triggered and docs:
        context = format_docs(docs)
        gen_start = time.time()
        try:
            result["answer"] = _answer_chain.invoke({"question": question, "rewritten_question": result["rewritten"], "context": context})
        except Exception as e:
            result["error"] = (result["error"] or "") + f" | 生成失败: {e}"
            result["answer"] = "【生成错误】"
        result["generation_time"] = round(time.time() - gen_start, 3)
    elif not docs and not fallback_triggered:
        result["answer"] = "【无检索结果】"
        result["generation_time"] = 0.0
    result["total_time"] = round(time.time() - start_time, 3)
    if docs and result["answer"] and not result["fallback"] and "【" not in result["answer"]:
        try:
            eval_prompt = f"""
你是一个严格的RAG评测专家。请对以下内容评分（0-10分）：
1. 检索相关性：检索到的文档是否包含回答问题的核心信息？
2. 答案忠实度：最终答案是否完全基于检索到的文档，无捏造？
3. 答案相关性：最终答案是否切中问题核心？

【用户问题】: {question}
【检索文档摘要】: {chr(10).join([f"- {d.page_content[:200]}..." for d in docs[:3]])}
【最终答案】: {result["answer"]}

只输出JSON：{{"retrieval_relevance":分数, "faithfulness":分数, "relevance":分数}}
"""
            eval_response = _judge_model.invoke(eval_prompt)
            json_match = re.search(r"\{.*\}", eval_response.content, re.DOTALL)
            if json_match:
                data = json.loads(json_match.group())
                result["retrieval_relevance"] = round(data.get("retrieval_relevance", 0), 1)
                result["faithfulness"] = round(data.get("faithfulness", 0), 1)
                result["relevance"] = round(data.get("relevance", 0), 1)
        except Exception as e:
            logging.getLogger("rag_app").warning(f"回复质量评测 LLM 评分失败 question={question[:40]}: {e}")
    if ground_truth and ground_truth.strip() and result["answer"] and not result["fallback"]:
        result["semantic_similarity"] = round(compute_semantic_similarity(_eval_embeddings, result["answer"], ground_truth), 3)
    return result


def _question_tokens(r: dict) -> int:
    """单题业务链路 token 总量（输入+输出，不含评测裁判）"""
    return sum(u.get("total_tokens", 0) for u in (r.get("llm_usage") or {}).values())


def _current_model_info(chunk_size: int | None = None, chunk_overlap: int = 0, cfg: dict | None = None) -> dict:
    """构造当前配置的 model_info（配置指纹的模型侧输入）。"""
    mcfg = dict(cfg or load_model_config())
    if cfg is None:
        mcfg["chat"] = _RT["cfg"].get("chat", mcfg.get("chat", {}))
        mcfg["embedding"] = _RT["cfg"].get("embedding", mcfg.get("embedding", {}))
    return build_model_info(
        mcfg,
        chunk_size=(
            chunk_size if chunk_size is not None
            else st.session_state.get("eval_chunk_size", st.session_state.get("chunk_size", 512))
        ),
        chunk_overlap=chunk_overlap,
        rerank_model=st.session_state.get("rerank_model", ""),
    )


# =========================================================
# 实验记录（本地 JSON 持久化）
# =========================================================
_EVAL_LOG_PATH = os.path.join(
    os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_experiments.json"
)


def _load_experiments() -> list:
    if not os.path.exists(_EVAL_LOG_PATH):
        return []
    try:
        with open(_EVAL_LOG_PATH, encoding="utf-8") as f:
            return json.load(f)
    except Exception:
        return []


def _save_experiment(record: dict) -> None:
    exps = _load_experiments()
    exps.append(record)
    try:
        with open(_EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(exps, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger("rag_app").warning(f"实验记录保存失败: {e}")


def _delete_experiment(exp_id: str) -> None:
    exps = [e for e in _load_experiments() if e.get("id") != exp_id]
    try:
        with open(_EVAL_LOG_PATH, "w", encoding="utf-8") as f:
            json.dump(exps, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logging.getLogger("rag_app").warning(f"实验记录删除失败: {e}")


def _build_experiment_record(eval_type: str, params: dict, metrics: dict) -> dict:
    return {
        "id": uuid.uuid4().hex[:12],
        "type": eval_type,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "params": params,
        "metrics": metrics,
    }


def _apply_to_production(exp: dict) -> None:
    """把实验的参数应用到主页面（线上配置）。"""
    p = exp.get("params", {})
    if "检索方式" in p and p["检索方式"]:
        st.session_state["retrieval_mode"] = p["检索方式"]
    if "top_k" in p and p["top_k"] is not None:
        st.session_state["top_k"] = p["top_k"]
    if "rerank" in p:
        st.session_state["rerank_enabled"] = bool(p["rerank"])
    if "置信度兜底" in p:
        st.session_state["confidence_fallback_enabled"] = bool(p["置信度兜底"])
    if "置信度阈值" in p and p["置信度阈值"] is not None:
        st.session_state["confidence_threshold"] = p["置信度阈值"]
    if "切块大小" in p and p["切块大小"]:
        st.session_state["chunk_size"] = p["切块大小"]


# =========================================================
# 页面主体（三步流程：数据 → 维度 → 开始；评测环境自动就绪）
# =========================================================

# ---------- Step 1: 评测环境（跟随全局配置，自动就绪）----------
eval_split_strategy = SPLIT_STRATEGIES[0]
eval_chunk_size = st.session_state.get("chunk_size", 512)
st.session_state["eval_split_strategy"] = eval_split_strategy
st.session_state["eval_chunk_size"] = eval_chunk_size
eval_model_cfg = _evaluation_cfg()

_eval_context_key = _evaluation_context_key(
    eval_model_cfg,
    eval_split_strategy,
    eval_chunk_size,
)

with st.container(border=True):
    st.markdown("#### 🧭 评测环境（自动跟随，无需配置）")
    _chat_cfg = eval_model_cfg.get("chat") or {}
    _chat_desc = _chat_cfg.get("model") or "未配置"
    st.caption(
        f"跟随全局配置：聊天模型 `{_chat_desc}` × 主页知识库。"
        "如需调整，请前往「模型设置」「知识库管理」页，本页将自动跟随。"
    )
    if st.session_state.get("eval_context_key") != _eval_context_key:
        try:
            with st.spinner("正在初始化评测环境..."):
                _build_evaluation_context(
                    _eval_context_key,
                    json.dumps(eval_model_cfg, ensure_ascii=False, sort_keys=True),
                    eval_split_strategy,
                    eval_chunk_size,
                )
            st.session_state["eval_context_key"] = _eval_context_key
        except Exception as e:
            st.error(f"评测环境初始化失败：{e}")
            if st.button("🔄 重试初始化", key="retry_eval_ctx"):
                _build_evaluation_context.clear()
                st.rerun()
    if st.session_state.get("eval_context_key") == _eval_context_key:
        st.success("✅ 评测环境已就绪")

# ---------- 加载评测上下文 ----------
_eval_context = None
if st.session_state.get("eval_context_key") == _eval_context_key:
    try:
        _eval_context = _build_evaluation_context(
            _eval_context_key,
            json.dumps(eval_model_cfg, ensure_ascii=False, sort_keys=True),
            eval_split_strategy,
            eval_chunk_size,
        )
    except Exception as e:
        st.error(f"评测环境加载失败：{e}")

# ---------- Step 2: 上传评测数据 ----------
with st.container(border=True):
    st.markdown("#### ① 上传评测数据")
    _render_eval_data_card()

# ---------- Step 3: 设置评测维度 ----------
with st.container(border=True):
    st.markdown("#### ② 设置评测维度")
    eval_type = st.radio(
        "评测类型",
        ["回复质量评测", "Agent 全链路评测"],
        horizontal=True,
        key="eval_type_select",
        label_visibility="collapsed",
    )
    _g1, _g2 = st.columns(2)
    with _g1:
        eval_mode = st.radio("检索方式", MODES, horizontal=True, key="eval_shared_mode")
        eval_rerank = st.checkbox("启用 Rerank 精排", key="eval_shared_rerank")
        eval_fb = st.checkbox("启用置信度兜底", value=True, key="eval_shared_fb", disabled=not eval_rerank)
    with _g2:
        eval_top_k = st.slider("Top-K", 1, 10, st.session_state.get("top_k", 5), key="eval_shared_topk")
        rerank_candidate_k = st.slider(
            "粗排候选数", 5, 30, 10, key="eval_shared_rerank_k",
            disabled=not eval_rerank,
        )
        eval_fb_th = st.slider(
            "置信度阈值", 0.0, 1.0, 0.5, 0.05, key="eval_shared_fb_th",
            disabled=not (eval_rerank and eval_fb),
        )
    if not eval_rerank:
        rerank_candidate_k = eval_top_k
        eval_fb_th = 0.5

# ---------- Step 4: 开始评测 ----------
_env_ready = _eval_context is not None
_data_ready = bool(st.session_state.get("eval_imported_samples"))
with st.container(border=True):
    st.markdown("#### ③ 开始评测")
    _check_env, _check_data, _check_dim = st.columns(3)
    with _check_env:
        st.markdown(f"{'✅' if _env_ready else '⬜'} **评测环境**")
    with _check_data:
        st.markdown(f"{'✅' if _data_ready else '⬜'} **评测数据**")
    with _check_dim:
        st.markdown("✅ **评测维度**")
    _samples_ready = _env_ready and _data_ready
    _run_btn = st.button(
        f"🚀 开始{eval_type}",
        type="primary",
        use_container_width=True,
        key="unified_run_btn",
        disabled=not _samples_ready,
    )

# ---------- 评测执行 + 结果 ----------
if _run_btn:
    collected = collect_questions()
    if collected is None:
        st.stop()
    questions, ground_truths, expected_intents = collected
    if not questions:
        st.warning("没有有效的问题。")
        st.stop()

    mode_label = f"{eval_mode}{' + Rerank' if eval_rerank else ''}"
    st.info(f"即将以【{mode_label}】对 **{len(questions)}** 个问题进行{eval_type}，请耐心等待...")
    _progress = st.progress(0)
    _status = st.empty()
    _time_est = st.empty()
    _t0 = time.time()
    _results = []
    _failed = 0

    for i, q in enumerate(questions):
        _status.text(f"正在评测 ({i+1}/{len(questions)}): {q[:40]}...")
        try:
            if eval_type == "回复质量评测":
                _results.append(run_eval_single(
                    q, ground_truths.get(q, ""), eval_mode, eval_rerank,
                    rerank_candidate_k, eval_top_k, eval_fb, eval_fb_th,
                    runtime=_eval_context,
                    similarity_threshold=st.session_state.get("similarity_threshold", 0.0),
                ))
            else:
                cfg = {
                    "mode": eval_mode, "rerank": eval_rerank, "rerank_k": rerank_candidate_k,
                    "fallback_on": eval_fb, "fallback_th": eval_fb_th,
                    "top_k": eval_top_k, "threshold": 0.0,
                }
                model_info = _current_model_info(chunk_size=eval_chunk_size, cfg=eval_model_cfg)
                fingerprint = config_fingerprint(cfg, model_info)
                _results.append(run_chain_eval_single(
                    _eval_context["deps"], q,
                    expected_intents.get(q, ""), ground_truths.get(q, ""), cfg,
                ))
        except Exception as e:
            _failed += 1
            _results.append({
                "question": q, "answer": f"【评测异常】{str(e)}", "doc_count": 0,
                "total_time": 0, "faithfulness": 0, "relevance": 0,
                "retrieval_relevance": 0, "semantic_similarity": 0, "error": str(e),
            })
        _elapsed = time.time() - _t0
        _time_est.text(f"⏱️ 已用 {_elapsed:.1f}s ｜ 预计剩余 {_elapsed/(i+1)*(len(questions)-i-1):.1f}s")
        _progress.progress((i + 1) / len(questions))

    _status.text("✅ 评测完成！")
    _time_est.empty()
    if _failed > 0:
        st.warning(f"⚠️ 有 {_failed} 个问题失败，已跳过。")
    st.success(f"✅ {eval_type}完成，共处理 {len(_results)} 个问题")

    # 记录实验
    _params = {
        "检索方式": eval_mode, "top_k": eval_top_k, "rerank": eval_rerank,
        "rerank_k": rerank_candidate_k, "置信度兜底": eval_fb, "置信度阈值": eval_fb_th,
        "切分方式": eval_split_strategy, "切块大小": eval_chunk_size,
        "聊天模型": (eval_model_cfg.get("chat") or {}).get("model", "未命名模型"),
        "嵌入模型": _RT["cfg"].get("embedding", {}).get("model", "bge-m3"),
    }
    if eval_type == "回复质量评测":
        _valid = [r for r in _results if r.get("doc_count", 0) > 0 and not r.get("fallback")]
        if _valid:
            _metrics = {
                "题数": len(_results),
                "忠实度": round(float(np.mean([r["faithfulness"] for r in _valid])), 2),
                "答案相关性": round(float(np.mean([r["relevance"] for r in _valid])), 2),
                "检索相关性": round(float(np.mean([r.get("retrieval_relevance", 0) for r in _valid])), 2),
                "语义相似度": round(float(np.mean([r.get("semantic_similarity", 0) for r in _valid])), 3),
                "平均耗时s": round(float(np.mean([r["total_time"] for r in _valid])), 3),
                "平均命中块": round(float(np.mean([r["doc_count"] for r in _valid])), 1),
                "兜底题数": sum(1 for r in _results if r.get("fallback")),
            }
        else:
            _metrics = {"题数": len(_results), "忠实度": 0, "答案相关性": 0, "检索相关性": 0, "语义相似度": 0, "平均耗时s": 0, "平均命中块": 0, "兜底题数": 0}
        _save_experiment(_build_experiment_record("回复质量评测", _params, _metrics))
        st.session_state["last_reply_results"] = _results
    else:
        _labeled = [r for r in _results if r.get("expected_intent") != "(未提供)"]
        _route_ok = sum(1 for r in _labeled if r.get("route_status") == "ok")
        _tot_tokens = sum(u.get("total_tokens", 0) for r in _results for u in (r.get("llm_usage") or {}).values())
        _metrics = {
            "题数": len(_results),
            "路由准确": f"{_route_ok}/{len(_labeled)}" if _labeled else "无标注",
            "总token": _tot_tokens,
            "平均耗时s": round(float(np.mean([r.get("total_time", 0) for r in _results])), 3) if _results else 0,
        }
        _save_experiment(_build_experiment_record("全链路评测", _params, _metrics))

    if eval_type == "Agent 全链路评测":
        st.session_state["last_chain_eval"] = {
            "results": _results, "cfg": cfg, "model_info": model_info,
            "fingerprint": fingerprint, "eval_context_key": _eval_context_key,
            "samples": [{"question": q, "expected_intent": expected_intents.get(q, ""), "ground_truth": ground_truths.get(q, "")} for q in questions],
        }

    st.rerun()

# ---------- 结果区（从 session_state 渲染） ----------
_last_reply = st.session_state.get("last_reply_results")
_last_chain = st.session_state.get("last_chain_eval")

if eval_type == "回复质量评测" and _last_reply:
    _results = _last_reply
    _valid = [r for r in _results if r.get("doc_count", 0) > 0 and not r.get("fallback")]
    if _valid:
        _avg_ret = float(np.mean([r["retrieval_time"] for r in _valid]))
        _avg_gen = float(np.mean([r["generation_time"] for r in _valid]))
        _avg_tot = float(np.mean([r["total_time"] for r in _valid]))
        _avg_f = float(np.mean([r["faithfulness"] for r in _valid]))
        _avg_r = float(np.mean([r["relevance"] for r in _valid]))
        _avg_rr = float(np.mean([r.get("retrieval_relevance", 0) for r in _valid]))
        _avg_sem = float(np.mean([r.get("semantic_similarity", 0) for r in _valid]))
        _avg_d = float(np.mean([r["doc_count"] for r in _valid]))
    else:
        _avg_ret = _avg_gen = _avg_tot = _avg_f = _avg_r = _avg_rr = _avg_sem = _avg_d = 0
    _fb_count = sum(1 for r in _results if r.get("fallback"))

    # ⚠️ 全军覆没时给明确提示，避免用户看着"全 0"困惑
    if _results and not _valid:
        _no_doc_ratio = sum(1 for r in _results if r.get("doc_count", 0) == 0) / len(_results)
        if _no_doc_ratio == 1.0:
            _err_msgs = [r.get("error") for r in _results if r.get("error")]
            _hint = "、".join(set(_err_msgs))[:200] if _err_msgs else "（无具体错误信息，可能检索被 silent 吞掉）"
            st.error(
                f"**所有 {_results and len(_results)} 个问题都未检索到任何文档。**\n\n"
                f"**可能原因**：\n"
                f"1. 检索方式选了「向量检索」但 Ollama 服务不可用\n"
                f"2. 嵌入模型配置错误或服务离线\n"
                f"3. 知识库为空（请确认主页 chroma_db 已有内容）\n\n"
                f"**建议**：\n"
                f"- 在「② 设置评测维度」把检索方式改为 **混合检索** 或 **关键词检索**（不依赖 Ollama）\n"
                f"- 或检查 Ollama 服务：`curl http://127.0.0.1:11434/api/tags`\n\n"
                f"**错误详情**：{_hint}"
            )

    _tab1, _tab2, _tab3 = st.tabs(["📊 指标总览", "📋 明细表", "🔍 逐题详情"])
    with _tab1:
        _metric_cards([
            ("检索时间", f"{_avg_ret:.2f}s", "召回文档的平均耗时"),
            ("生成时间", f"{_avg_gen:.2f}s", "LLM 生成答案的平均耗时"),
            ("总耗时", f"{_avg_tot:.2f}s", "单题端到端平均耗时"),
            ("平均命中块", f"{_avg_d:.1f} 块", "每题平均召回文档块数"),
        ])
        _metric_cards([
            ("忠实度", f"{_avg_f:.1f}/10", "答案是否基于检索文档、无编造"),
            ("答案相关性", f"{_avg_r:.1f}/10", "答案是否切中问题核心"),
            ("检索相关性", f"{_avg_rr:.1f}/10", "检索文档是否含核心信息"),
            ("语义相似度", f"{_avg_sem:.2f}", "与标准答案的向量相似度"),
        ])
        if eval_rerank and eval_fb:
            st.caption(f"🛟 置信度兜底触发 {_fb_count} 题（不参与质量评分）")
        _csv_df = pd.DataFrame(_results).copy()
        if "docs" in _csv_df.columns:
            _csv_df["docs"] = _csv_df["docs"].apply(lambda d: len(d) if isinstance(d, list) else 0)
        _csv = _csv_df.to_csv(index=False).encode("utf-8-sig")
        st.download_button("📥 下载评测结果 (CSV)", _csv,
                           file_name=f"rag_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with _tab2:
        _df = pd.DataFrame(_results).copy()
        _df["answer"] = _df["answer"].apply(lambda x: str(x)[:100] + "..." if len(str(x)) > 100 else x)
        _cols = ["question", "answer", "doc_count", "retrieval_time", "total_time",
                 "faithfulness", "relevance", "retrieval_relevance", "semantic_similarity"]
        if "ground_truth" in _df.columns:
            _cols.insert(1, "ground_truth")
        st.dataframe(_df[_cols], use_container_width=True)
    with _tab3:
        for idx, r in enumerate(_results):
            with st.expander(f"题{idx+1}：{r['question'][:50]} ｜ 忠实度 {r['faithfulness']}"):
                if r.get("ground_truth"):
                    st.caption(f"📋 标准答案：{r['ground_truth']}")
                st.caption(f"✏️ 改写后：{r.get('rewritten', 'N/A')}")
                st.write(r["answer"])
                if r.get("docs"):
                    with st.expander(f"检索到的文档块（{len(r['docs'])} 个）"):
                        for i, doc in enumerate(r["docs"][:3]):
                            st.text(f"块{i+1}: {doc.page_content[:300]}...")
                _m1, _m2, _m3 = st.columns(3)
                with _m1:
                    st.metric("检索耗时", f"{r['retrieval_time']}s")
                with _m2:
                    st.metric("总耗时", f"{r['total_time']}s")
                with _m3:
                    st.metric("命中块", r['doc_count'])
                if r.get("error"):
                    st.error(f"错误: {r['error']}")

elif eval_type == "Agent 全链路评测" and _last_chain:
    _results = _last_chain["results"]
    cfg = _last_chain["cfg"]
    model_info = _last_chain.get("model_info")
    fingerprint = _last_chain.get("fingerprint")
    _n = len(_results)
    _labeled = [r for r in _results if r.get("expected_intent") != "(未提供)"]
    _rc = sum(1 for r in _labeled if r["route_status"] == "ok")
    _rag = [r for r in _results if "rag_retrieve" in r.get("node_names", [])]
    _rh = sum(1 for r in _rag if r["retrieve_status"] == "ok")
    _rf = sum(1 for r in _rag if r["retrieve_status"] == "warn")
    _rm = sum(1 for r in _rag if r["retrieve_status"] == "error")
    _tool = [r for r in _results if r.get("tool_status") is not None]
    _cl = sum(1 for r in _tool if "澄清" in (r["tool_detail"] or ""))
    _ab = [r for r in _results if r["attribution"] != "✅ 正常"]

    def _avg_s(key):
        vals = [r["scores"][key] for r in _results if isinstance(r.get("scores"), dict) and r["scores"].get(key) is not None]
        return float(np.mean(vals)) if vals else 0.0

    def _sum_t(key):
        return sum(u.get(key, 0) for r in _results for u in (r.get("llm_usage") or {}).values())

    _ti, _to, _tc = _sum_t("input_tokens"), _sum_t("output_tokens"), _sum_t("calls")

    _tab1, _tab2, _tab3, _tab4 = st.tabs(["📊 指标总览", "🔬 节点诊断", "📋 明细表", "🔍 逐题详情"])
    with _tab1:
        _metric_cards([
            ("路由准确率", f"{_rc}/{len(_labeled)}" if _labeled else "无标注", "意图路由正确的比例"),
            ("检索命中率", f"{_rh}/{len(_rag)}" if _rag else "—", f"兜底{_rf}，无命中{_rm}"),
            ("兜底+无命中", f"{_rf+_rm}/{len(_rag)}" if _rag else "—", "置信度兜底+无命中+异常"),
            ("工具澄清率", f"{_cl}/{len(_tool)}" if _tool else "—", "参数不足主动澄清"),
            ("异常归因题数", f"{len(_ab)}/{_n}", "被归因到某节点的题数"),
        ], cols=5)
        _metric_cards([
            ("路由合理性", f"{_avg_s('route_score'):.1f}/10", "意图识别与参数提取"),
            ("改写质量", f"{_avg_s('rewrite_score'):.1f}/10", "改写后问题是否保留核心语义"),
            ("检索相关性", f"{_avg_s('retrieval_score'):.1f}/10", "检索文档是否含核心信息"),
            ("答案忠实度", f"{_avg_s('faithfulness'):.1f}/10", "答案是否基于文档、无编造"),
            ("回复相关性", f"{_avg_s('relevance'):.1f}/10", "最终回复是否切中问题"),
        ], cols=5)
        _metric_cards([
            ("总输入tokens", f"{_ti:,}", "业务链路输入合计"),
            ("总输出tokens", f"{_to:,}", "业务链路输出合计"),
            ("平均每题tokens", f"{(_ti+_to)/_n:,.0f}" if _n else "—", "每题平均消耗"),
            ("LLM调用次数", f"{_tc}", "不含评测裁判"),
        ])
        _csv_rows = []
        for r in _results:
            sc = r.get("scores", {})
            _csv_rows.append({
                "question": r["question"], "expected_intent": r.get("expected_intent", ""),
                "predicted_intent": r.get("predicted_intent", ""), "route_status": r.get("route_status", ""),
                "rewritten": r.get("rewritten", ""), "answer": r.get("answer", ""),
                "doc_count": len(r.get("docs", [])), "total_time": r.get("total_time", 0),
                "route_score": sc.get("route_score", ""), "faithfulness": sc.get("faithfulness", ""),
                "relevance": sc.get("relevance", ""), "attribution": r.get("attribution", ""),
            })
        st.download_button("📥 下载全链路结果 (CSV)",
                           pd.DataFrame(_csv_rows).to_csv(index=False).encode("utf-8-sig"),
                           file_name=f"chain_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                           mime="text/csv")
    with _tab2:
        _lat_rows = [{"节点": NODE_LABELS.get(nr["node"], nr["node"]), "耗时s": nr["latency"]}
                      for r in _results for nr in r.get("node_runs", [])]
        if _lat_rows:
            _lat = pd.DataFrame(_lat_rows).groupby("节点")["耗时s"].mean().sort_values(ascending=False)
            st.bar_chart(_lat)
            st.dataframe(_lat.rename("平均耗时s").round(3).to_frame(), use_container_width=True)
        _attr = pd.Series([r["attribution"] for r in _results]).value_counts()
        st.bar_chart(_attr)
        st.dataframe(_attr.rename("题数").to_frame(), use_container_width=True)
        _tok_rows = [{"节点": NODE_LABELS.get(n, n), "输入": u.get("input_tokens", 0), "输出": u.get("output_tokens", 0)}
                      for r in _results for n, u in (r.get("llm_usage") or {}).items()]
        if _tok_rows:
            _tk = pd.DataFrame(_tok_rows).groupby("节点").sum()
            st.bar_chart(_tk[["输入", "输出"]])
            st.dataframe(_tk, use_container_width=True)
    with _tab3:
        _rows = []
        for r in _results:
            sc = r.get("scores", {})
            _rows.append({
                "问题": r["question"][:30], "期望意图": r.get("expected_intent", ""),
                "实际意图": r.get("predicted_intent", ""),
                "路由": STATUS_ICON.get(r.get("route_status"), "—"),
                "检索": STATUS_ICON.get(r.get("retrieve_status"), "—"),
                "回复": STATUS_ICON.get(r.get("answer_status"), "—"),
                "命中块": len(r.get("docs", [])), "忠实度": sc.get("faithfulness", ""),
                "相关性": sc.get("relevance", ""), "耗时s": round(r.get("total_time", 0), 2),
                "tokens": _question_tokens(r), "归因": r.get("attribution", ""),
            })
        st.dataframe(pd.DataFrame(_rows), use_container_width=True, hide_index=True)
    with _tab4:
        for idx, r in enumerate(_results):
            _icon = "✅" if r["attribution"] == "✅ 正常" else ("❌" if r.get("graph_error") or "异常" in r["attribution"] else "⚠️")
            with st.expander(f"{_icon} 题{idx+1}：{r['question'][:50]} ｜ {r.get('predicted_intent','?')} ｜ {r.get('total_time',0)}s"):
                _lat_map = {nr["node"]: nr["latency"] for nr in r.get("node_runs", [])}
                _chips = ["📥 进线", f"🧭 路由{STATUS_ICON.get(r.get('route_status'),'')}（{_lat_map.get('intent',0)}s）"]
                if "rag_retrieve" in r.get("node_names", []):
                    _chips.append(f"✏️ 改写{STATUS_ICON.get(r.get('rewrite_status'),'')}")
                    _chips.append(f"🔍 检索{STATUS_ICON.get(r.get('retrieve_status'),'')}（{_lat_map.get('rag_retrieve',0)}s）")
                if "order" in r.get("node_names", []):
                    _chips.append(f"🛒 查订单{STATUS_ICON.get(r.get('tool_status'),'')}（{_lat_map.get('order',0)}s）")
                if "ticket" in r.get("node_names", []):
                    _chips.append(f"🎫 建工单{STATUS_ICON.get(r.get('tool_status'),'')}（{_lat_map.get('ticket',0)}s）")
                if "rag_answer" in r.get("node_names", []):
                    _chips.append(f"💬 回复{STATUS_ICON.get(r.get('answer_status'),'')}（{_lat_map.get('rag_answer',0)}s）")
                if "chitchat" in r.get("node_names", []):
                    _chips.append(f"💬 闲聊{STATUS_ICON.get(r.get('answer_status'),'')}（{_lat_map.get('chitchat',0)}s）")
                st.markdown("　➡️　".join(_chips))
                _usage = r.get("llm_usage") or {}
                if _usage:
                    st.caption("📮 Token（输入→输出）：" + "　".join(f"{NODE_SHORT.get(k,k)} {v.get('input_tokens',0)}→{v.get('output_tokens',0)}" for k, v in _usage.items()))
                if r.get("graph_error"):
                    st.error(f"图执行异常：{r['graph_error']}")
                st.caption(f"🧭 {r.get('route_status_text','')} ｜ 参数：{r.get('tool_args') or '无'} ｜ 归因：{r['attribution']}")
                if r.get("rewritten"):
                    st.caption(f"✏️ 原问题：{r['question']}")
                    st.write(f"改写后：{r['rewritten']}")
                if r.get("retrieve_status") is not None:
                    st.caption(f"🔍 {STATUS_ICON.get(r.get('retrieve_status'),'')} {r.get('retrieve_detail','')}")
                    if r.get("docs"):
                        for di, doc in enumerate(r["docs"][:5], start=1):
                            meta = doc.metadata or {}
                            st.write(f"- 【{meta.get('source','?')}·块{meta.get('chunk_id','?')}】{doc.page_content[:80].replace(chr(10),' ')}...")
                if r.get("tool_status") is not None:
                    st.caption(f"🛠️ {STATUS_ICON.get(r.get('tool_status'),'')} {r.get('tool_detail','')}")
                st.write(r.get("answer") or "（无回复）")
                sc = r.get("scores", {})
                if sc:
                    st.caption("裁判评分：" + "　".join(f"{k}={v}" for k, v in sc.items()) + f" ｜ 语义相似度：{r.get('semantic_similarity',0)}")
    # LangSmith 上报（折叠在结果区底部）
    with st.expander("☁️ 高级能力：同步到 LangSmith（可选）", expanded=False):
        _ls_key = st.session_state.get("ls_api_key", "")
        _can = is_langsmith_ready() or bool(_ls_key)
        if not _can:
            st.caption("填写 LANGSMITH_API_KEY 后即可上报（smith.langchain.com 免费注册）。")
        else:
            _ls_ds = st.text_input("数据集名", value="客服Agent全链路评测", key="ls_dataset")
            _ls_pf = st.text_input("实验名前缀", value="chain-eval", key="ls_prefix")
            if st.button("☁️ 上报为 LangSmith 实验", key="ls_upload_btn"):
                if _ls_key:
                    set_api_key(_ls_key)
                with st.spinner("正在同步并上报..."):
                    try:
                        if not model_info or not fingerprint:
                            model_info = _current_model_info(chunk_size=eval_chunk_size, cfg=eval_model_cfg)
                            fingerprint = config_fingerprint(cfg, model_info)
                        _upload_deps = (
                            _eval_context["deps"]
                            if _eval_context is not None and _last_chain.get("eval_context_key") == _eval_context_key
                            else CHAIN_EVAL_DEPS
                        )
                        _info = upload_chain_experiment(
                            _upload_deps, cfg, _last_chain["samples"],
                            results_by_question={fingerprint: {r["question"]: r for r in _results}},
                            model_info=model_info,
                            dataset_name=_ls_ds, experiment_prefix=_ls_pf,
                        )
                        st.success(f"✅ 已上报：{_info['experiment_name']}（{_info['n']} 个样本）")
                        st.markdown(f"🔗 [打开实验看板]({_info['url']})")
                    except Exception as e:
                        st.error(f"LangSmith 上报失败：{e}（本地评测结果不受影响）")

# =========================================================
# 实验记录（折叠）
# =========================================================
_exps = _load_experiments()
with st.expander(f"📋 实验记录（{len(_exps)}）", expanded=False):
    if not _exps:
        st.caption("暂无记录，跑一次评测后会自动留档。")
    else:
        _exp_by_id = {e["id"]: e for e in _exps}
        _sel = st.multiselect(
            "选择 2~4 条对比",
            options=list(reversed([e["id"] for e in _exps])),
            max_selections=4,
            format_func=lambda eid: f"{_exp_by_id[eid].get('created_at','')} ｜ {_exp_by_id[eid].get('type','')}",
            key="compare_experiment_ids",
        )
        if len(_sel) >= 2:
            _cmp = {}
            for eid in _sel:
                _e = _exp_by_id[eid]
                _label = f"{_e.get('created_at','')}\n{_e.get('type','')}"
                _flat = {f"参数·{k}": v for k, v in (_e.get("params") or {}).items()}
                _flat.update({f"指标·{k}": v for k, v in (_e.get("metrics") or {}).items()})
                _cmp[_label] = _flat
            st.dataframe(pd.DataFrame(_cmp), use_container_width=True)

        for _exp in reversed(_exps):
            _p = _exp.get("params", {})
            _m = _exp.get("metrics", {})
            if _exp.get("type") == "回复质量评测":
                _summary = f"{_exp['created_at']} ｜ 回复质量 ｜ {_p.get('检索方式','')} · 忠实度 {_m.get('忠实度','—')} ｜ 检索相关 {_m.get('检索相关性','—')}"
            else:
                _summary = f"{_exp['created_at']} ｜ 全链路 ｜ {_p.get('检索方式','')} · 路由 {_m.get('路由准确','—')} ｜ token {_m.get('总token',0)}"
            with st.expander(_summary):
                _c1, _c2 = st.columns(2)
                with _c1:
                    st.caption("参数")
                    for k, v in _p.items():
                        st.caption(f"{k}：{v}")
                with _c2:
                    st.caption("指标")
                    for k, v in _m.items():
                        st.caption(f"{k}：{v}")
                _b1, _b2 = st.columns(2)
                with _b1:
                    if st.button("✅ 应用为线上配置", key=f"apply_{_exp['id']}", use_container_width=True):
                        _apply_to_production(_exp)
                        st.success("✅ 已应用（模型请到模型设置页切换）")
                        st.rerun()
                with _b2:
                    if st.button("🗑️ 删除", key=f"del_{_exp['id']}", use_container_width=True):
                        _delete_experiment(_exp["id"])
                        st.rerun()
