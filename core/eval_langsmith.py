# -*- coding: utf-8 -*-
"""LangSmith 评测上报层（框架无关）

把「全链路节点级评测」（core/chain_eval.py，自有评分逻辑）的结果上报到
LangSmith 平台，获得专业实验看板：跨实验对比、按指标筛选、自定义图表。

设计要点：
  1. 评分逻辑完全复用自有 chain_eval.run_chain_eval_single —— LangSmith 只负责
     存储/展示/对比，不在平台侧重算，看板数字与本地 Streamlit 看板完全一致。
  2. 优先复用本地已跑出的结果（results_by_question 缓存），evaluate() 不重复
     消耗本地 LLM 调用；数据集中有而本地未跑的样本才实时执行。
  3. 自有评分被封装为 LangSmith evaluators（feedback 指标）：路由正确/裁判各项
     评分/语义相似度/耗时/命中块数/归因节点，可在看板上自由筛选与建图。
  4. 优雅降级：未装 SDK 或未配置 LANGSMITH_API_KEY 时 is_langsmith_ready()=False，
     本地评测功能不受任何影响。

⚠️ 数据合规：上报内容包含问题文本、最终回答、命中文档摘要、工具参数等，
   会上传至 LangSmith SaaS（smith.langchain.com，用户自己的账号空间）。

环境变量（.env 或页面内输入 Key）：
  LANGSMITH_API_KEY      必填，https://smith.langchain.com → Settings → API Keys
  LANGSMITH_ENDPOINT     可选，默认 https://api.smith.langchain.com
  LANGSMITH_TRACING      可选，置 true 后线上对话的链/图调用也会自动 trace
"""
import hashlib
import json
import os
from datetime import datetime

from core.chain_eval import run_chain_eval_single, NODE_LABELS


def is_langsmith_ready() -> bool:
    """SDK 已安装且配置了 API Key 才具备上报条件"""
    try:
        import langsmith  # noqa: F401
    except ImportError:
        return False
    return bool(os.environ.get("LANGSMITH_API_KEY"))


def set_api_key(api_key: str, endpoint: str = "") -> None:
    """页面内临时设置 API Key（写入进程环境变量，不持久化到 .env）"""
    api_key = (api_key or "").strip()
    if api_key:
        os.environ["LANGSMITH_API_KEY"] = api_key
    endpoint = (endpoint or "").strip()
    if endpoint:
        os.environ["LANGSMITH_ENDPOINT"] = endpoint


# 全链路各图节点名（与 chain_eval._NodeTokenCollector.NODE_NAMES 一致）
# 用于把每节点耗时 / token 拆分为独立 feedback 指标（latency_<node> / tokens_<node>）
NODE_NAMES = ("intent", "rag_retrieve", "rag_answer", "order", "ticket", "chitchat")


def _model_identity(section: dict, local_label: str) -> str:
    """把模型配置段（chat / embedding）转成稳定标识字符串。
    本地模型直接用模型名；API 模型拼 provider + model，避免不同厂商同名混淆。"""
    section = section or {}
    if section.get("source") == "api":
        return f"api:{section.get('provider', '')}:{section.get('model', '')}"
    return section.get("model") or local_label


def build_model_info(
    model_cfg: dict,
    chunk_size: int | None = None,
    chunk_overlap: int | None = None,
    rerank_model: str | None = None,
) -> dict:
    """从模型配置 + 分块 / rerank 参数构造 model_info（配置指纹的模型侧输入）。

    model_cfg 即 model_factory.load_model_config() 的返回值；chunk_size / chunk_overlap /
    rerank_model 来自页面当前参数。任何字段缺失都不会抛异常（config_fingerprint 内兜底）。
    """
    model_cfg = model_cfg or {}
    chat = model_cfg.get("chat", {}) or {}
    emb = model_cfg.get("embedding", {}) or {}
    from core import model_factory  # 延迟导入，避免循环依赖；仅取常量

    chat_label = _model_identity(chat, model_factory.LOCAL_CHAT_MODEL)
    emb_label = _model_identity(emb, model_factory.LOCAL_EMBEDDING_MODEL)
    return {
        # 嵌入模型 / 回复模型 / 改写模型（改写与回复当前共用聊天模型，字段分开以便未来拆分）
        "embedding_model": emb_label,
        "answer_model": chat_label,
        "rewrite_model": chat_label,
        # 分块策略（决定知识库切块，间接影响检索召回）
        "chunk_size": chunk_size,
        "chunk_overlap": chunk_overlap,
        # Rerank 模型（精排质量）
        "rerank_model": rerank_model,
        # 知识库向量空间标识：换嵌入模型后必须重建库，指纹必须随之变化避免新旧库混比
        "kb_embedding_space": model_cfg.get("kb_embedding_space") or emb_label,
    }


def config_fingerprint(cfg: dict, model_info: dict | None = None) -> str:
    """把影响全链路结果的配置 + 模型标识拼成稳定哈希，作为缓存隔离 key。

    指纹变化 → 缓存失效自动重跑；指纹不变 → 复用旧结果。
    cfg 包含：mode/rerank/rerank_k/fallback_on/fallback_th/top_k/threshold。
    model_info 见 build_model_info（嵌入/回复/改写模型、分块、rerank 模型、向量空间）。
    所有字段都有默认兜底，缺字段不会抛异常导致评测中断。
    """
    cfg = cfg or {}
    model_info = model_info or {}
    # 逐字段显式取值 + 默认值，保证序列化稳定、可复现
    payload = {
        "mode": cfg.get("mode", ""),
        "rerank": bool(cfg.get("rerank", False)),
        "rerank_k": cfg.get("rerank_k", 0),
        "fallback_on": bool(cfg.get("fallback_on", True)),
        "fallback_th": cfg.get("fallback_th", 0.0),
        "top_k": cfg.get("top_k", 5),
        "threshold": cfg.get("threshold", 0.0),
        "embedding_model": model_info.get("embedding_model", ""),
        "answer_model": model_info.get("answer_model", ""),
        "rewrite_model": model_info.get("rewrite_model", ""),
        "chunk_size": model_info.get("chunk_size"),
        "chunk_overlap": model_info.get("chunk_overlap"),
        "rerank_model": model_info.get("rerank_model", ""),
        "kb_embedding_space": model_info.get("kb_embedding_space", ""),
    }
    raw = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.md5(raw.encode("utf-8")).hexdigest()


def _safe_label(text: str) -> str:
    """实验名 / 标签清理：去掉空格与斜杠，避免实验名里出现非法分隔符。"""
    return str(text or "").strip().replace(" ", "-").replace("/", "-") or "default"


def _is_legacy_cache(results_by_question: dict) -> bool:
    """判断缓存结构：旧扁平 {question: result} 还是新嵌套 {fingerprint: {question: result}}。

    旧结构 value 是 chain_eval 结果对象（含 scores / node_runs 字段）；
    新结构 value 是 {question: result} 映射。用于向后兼容识别。
    """
    if not results_by_question:
        return False
    first_value = next(iter(results_by_question.values()))
    return isinstance(first_value, dict) and (
        "scores" in first_value or "node_runs" in first_value
    )


def sync_dataset(client, dataset_name: str, samples: list[dict]) -> str:
    """幂等同步数据集：不存在则创建；按 question 去重补充样本。
    samples: [{"question", "expected_intent"(可选), "ground_truth"(可选)}]
    返回 dataset_id。
    """
    description = (
        "智能客服 Agent 全链路评测集（节点诊断）：进线→路由→改写→检索→工具→回复。"
        "reference 字段 expected_intent/ground_truth 为可选标注。"
    )
    if client.has_dataset(dataset_name=dataset_name):
        dataset = client.read_dataset(dataset_name=dataset_name)
    else:
        dataset = client.create_dataset(dataset_name, description=description)

    existing = {
        (ex.inputs or {}).get("question")
        for ex in client.list_examples(dataset_id=dataset.id)
    }
    for s in samples:
        q = (s.get("question") or "").strip()
        if not q or q in existing:
            continue
        client.create_example(
            inputs={"question": q},
            outputs={
                "expected_intent": s.get("expected_intent") or "",
                "ground_truth": s.get("ground_truth") or "",
            },
            dataset_id=dataset.id,
        )
    return dataset.id


def _flatten_result(r: dict) -> dict:
    """把 chain_eval 结果拍平为 LangSmith run outputs（看板列/feedback 数据源）"""
    sc = r.get("scores") or {}
    flat = {
        "answer": r.get("answer", ""),
        "predicted_intent": r.get("predicted_intent", ""),
        "expected_intent": r.get("expected_intent", ""),
        "node_chain": " → ".join(
            NODE_LABELS.get(n, n) for n in r.get("node_names", [])
        ),
        "rewritten": r.get("rewritten", ""),
        "route_note": r.get("route_note", ""),
        "tool_args": str(r.get("tool_args") or {}),
        # 节点状态（ok/warn/error）
        "route_status": r.get("route_status"),
        "rewrite_status": r.get("rewrite_status"),
        "retrieve_status": r.get("retrieve_status"),
        "tool_status": r.get("tool_status"),
        "answer_status": r.get("answer_status"),
        # 数值指标
        "route_correct": r.get("route_status") == "ok",
        "route_score": sc.get("route_score"),
        "rewrite_score": sc.get("rewrite_score"),
        "retrieval_score": sc.get("retrieval_score"),
        "faithfulness": sc.get("faithfulness"),
        "relevance": sc.get("relevance"),
        "tool_score": sc.get("tool_score"),
        "semantic_similarity": r.get("semantic_similarity", 0),
        "total_time": r.get("total_time", 0),
        "doc_count": len(r.get("docs") or []),
        # 归因（类别型 feedback）
        "attribution": r.get("attribution", ""),
        "graph_error": r.get("graph_error") or "",
    }
    # ---- 节点级耗时：node_runs 的每节点 latency 拆为 latency_<node> ----
    # 只在实际经过该节点的结果里输出该字段，未经过的节点不输出 → evaluator 跳过不打分
    for nr in r.get("node_runs") or []:
        node = (nr or {}).get("node")
        if node:
            flat[f"latency_{node}"] = nr.get("latency")
    # ---- 节点级 token：llm_usage 的每节点 total_tokens 拆为 tokens_<node> ----
    for node, usage in (r.get("llm_usage") or {}).items():
        if isinstance(usage, dict):
            flat[f"tokens_{node}"] = usage.get("total_tokens")
    return flat


# ---------- evaluator 工厂：把自有评分上报为 LangSmith feedback 指标 ----------

def _score_evaluator(output_key: str, label: str):
    """连续分指标（0-10 裁判分 / 相似度 / 耗时 / 块数）；不适用该分支时跳过"""
    def evaluator(run, example):
        from langsmith.evaluation import EvaluationResult

        outputs = getattr(run, "outputs", None) or {}
        val = outputs.get(output_key)
        if val is None:
            return None  # 该分支无此指标（如工具分支无检索分），跳过不打分
        return EvaluationResult(key=label, score=float(val))

    evaluator.__name__ = f"eval_{output_key}"
    return evaluator


def _route_correct_evaluator(run, example):
    """二值指标：路由是否正确（有 expected_intent 标注时有效）"""
    from langsmith.evaluation import EvaluationResult

    outputs = getattr(run, "outputs", None) or {}
    if outputs.get("expected_intent") in ("", "(未提供)", None):
        return None
    return EvaluationResult(
        key="路由正确", score=1.0 if outputs.get("route_correct") else 0.0
    )


def _attribution_evaluator(run, example):
    """类别指标：错误归因节点（看板可按该字段筛选/分组）"""
    from langsmith.evaluation import EvaluationResult

    outputs = getattr(run, "outputs", None) or {}
    attr = outputs.get("attribution") or "未知"
    return EvaluationResult(key="归因节点", value=attr, score=None)


def build_evaluators() -> list:
    """构造全部 evaluators（评分逻辑来自本地，LangSmith 仅记录展示）"""
    evals = [
        _route_correct_evaluator,
        _score_evaluator("route_score", "裁判-路由合理性"),
        _score_evaluator("rewrite_score", "裁判-改写质量"),
        _score_evaluator("retrieval_score", "裁判-检索相关性"),
        _score_evaluator("faithfulness", "裁判-答案忠实度"),
        _score_evaluator("relevance", "裁判-回复相关性"),
        _score_evaluator("tool_score", "裁判-工具执行"),
        _score_evaluator("semantic_similarity", "语义相似度"),
        _score_evaluator("total_time", "总耗时(秒)"),
        _score_evaluator("doc_count", "命中块数"),
        _attribution_evaluator,
    ]
    # 节点级指标：每节点耗时 / token 作为独立 feedback 上报；
    # 未经过该分支的节点在 outputs 里没有对应字段 → _score_evaluator 返回 None 跳过
    for node in NODE_NAMES:
        evals.append(_score_evaluator(f"latency_{node}", f"耗时-{node}"))
        evals.append(_score_evaluator(f"tokens_{node}", f"Token-{node}"))
    return evals


def upload_chain_experiment(
    deps: dict,
    cfg: dict,
    samples: list[dict],
    results_by_question: dict | None = None,
    model_info: dict | None = None,
    dataset_name: str = "客服Agent全链路评测",
    experiment_prefix: str = "chain-eval",
) -> dict:
    """把全链路评测结果上报为一次 LangSmith 实验。

    参数：
      deps: {"graph", "chat_model", "embeddings"}（同 chain_eval.run_chain_eval_single）
      cfg:  {"mode","rerank","rerank_k","fallback_on","fallback_th","top_k","threshold"}
      samples: [{"question","expected_intent","ground_truth"}] 数据集样本
      results_by_question: 本地已跑完的缓存，支持两种结构：
          - 新结构（推荐）：{config_fingerprint: {question: chain_eval 结果}}
          - 旧扁平结构：{question: chain_eval 结果}（无指纹信息，会按当前配置重跑）
      model_info: 见 build_model_info（用于配置指纹与实验命名）；缺省时按空信息兜底
      dataset_name / experiment_prefix: LangSmith 上的数据集名 / 实验名前缀
    返回：{"experiment_name", "url", "comparison_url", "dataset_name", "n",
          "config_fingerprint", "legacy_cache_dropped"}
    """
    if not is_langsmith_ready():
        raise RuntimeError(
            "LangSmith 未就绪：请先在页面填写 LANGSMITH_API_KEY"
            "（或在 .env 中配置），Key 可在 smith.langchain.com 免费申请。"
        )
    from langsmith import Client, evaluate

    # ---- 配置指纹：决定当前上报属于哪个配置分组，命中同分组缓存才复用 ----
    fp = config_fingerprint(cfg, model_info)

    # ---- 缓存规范化：统一为 {fingerprint: {question: result}} ----
    cache = results_by_question or {}
    legacy_dropped = False
    if _is_legacy_cache(cache):
        # 旧扁平缓存没有指纹信息，无法判断属于哪个配置，直接丢弃按当前配置重跑，
        # 避免把旧配置结果错配到新配置上（宁可重跑，也不产出不可比的数据）
        legacy_dropped = True
        cache = {}
    # 当前指纹分组下已跑过的样本（fingerprint 不存在或值非 dict 时视为空，兜底重跑）
    per_fp = cache.get(fp) if isinstance(cache.get(fp), dict) else {}

    samples_by_q = {s["question"]: s for s in samples}

    client = Client()  # 从环境变量读取 LANGSMITH_API_KEY / LANGSMITH_ENDPOINT
    sync_dataset(client, dataset_name, samples)

    def target(inputs: dict) -> dict:
        """evaluate 驱动的目标函数：优先复用当前配置指纹下的缓存，未跑过的实时执行"""
        q = inputs["question"]
        ref = samples_by_q.get(q, {})
        r = per_fp.get(q)
        if r is None:
            r = run_chain_eval_single(
                deps, q,
                ref.get("expected_intent", ""), ref.get("ground_truth", ""), cfg
            )
        return _flatten_result(r)

    # ---- 实验命名带配置标签：便于在 LangSmith 看板按名称筛选出对比组 ----
    mi = model_info or {}
    chunk_val = mi.get("chunk_size")
    chunk_tag = f"chunk{chunk_val}" if chunk_val is not None else "chunkna"
    mode_tag = _safe_label(cfg.get("mode", ""))
    model_tag = _safe_label(mi.get("answer_model", ""))
    ts = datetime.now().strftime("%Y%m%d-%H%M%S")
    experiment_name = f"{experiment_prefix}-{chunk_tag}-{mode_tag}-{model_tag}-{ts}"

    results = evaluate(
        target,
        data=dataset_name,
        evaluators=build_evaluators(),
        experiment=experiment_name,
        max_concurrency=1,  # 本地 Ollama 串行，避免并发打爆模型服务
        client=client,
        description=(
            f"全链路节点级评测｜检索={cfg.get('mode')} rerank={cfg.get('rerank')} "
            f"top_k={cfg.get('top_k')}｜回复模型={model_tag}｜分块={chunk_val}｜"
            f"评分逻辑来自本地 chain_eval，看板可跨实验对比"
        ),
        metadata={
            "retrieval_mode": cfg.get("mode"),
            "rerank_enabled": cfg.get("rerank"),
            "rerank_candidates": cfg.get("rerank_k"),
            "confidence_fallback": cfg.get("fallback_on"),
            "confidence_threshold": cfg.get("fallback_th"),
            "top_k": cfg.get("top_k"),
            "source": "streamlit-chain-eval",
            # 模型侧标识（配合实验名做跨实验筛选）
            "embedding_model": mi.get("embedding_model", ""),
            "answer_model": mi.get("answer_model", ""),
            "rewrite_model": mi.get("rewrite_model", ""),
            "chunk_size": mi.get("chunk_size"),
            "chunk_overlap": mi.get("chunk_overlap"),
            "rerank_model": mi.get("rerank_model", ""),
            "kb_embedding_space": mi.get("kb_embedding_space", ""),
            "config_fingerprint": fp,
        },
    )

    return {
        "experiment_name": getattr(results, "experiment_name", experiment_name),
        "url": getattr(results, "url", ""),
        "comparison_url": getattr(results, "comparison_url", ""),
        "dataset_name": dataset_name,
        "n": len(samples),
        "config_fingerprint": fp,
        "legacy_cache_dropped": legacy_dropped,
    }
