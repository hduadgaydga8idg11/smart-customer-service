# -*- coding: utf-8 -*-
"""全量对比评测基础库
- 被测模型：deepseek-chat（生成/路由，来自模型设置页当前配置）
- 裁判模型：deepseek-reasoner（独立于被测，只做评分）
- L0 硬指标：意图路由、事实覆盖率(Recall)、工具参数、兜底触发 —— 零模型偏差
- L1 裁判分：检索相关性/忠实度/答案相关性，锚点化 rubric
所有组都走【真实生产图】(build_agent_graph)，参数经 initial_state 透传。
"""
import csv
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import jieba
from dotenv import load_dotenv
from langchain_chroma import Chroma
from langchain_core.output_parsers import StrOutputParser

from core import retrieval as R
from core.agent_graph import build_agent_graph
from core.chain_eval import normalize_intent
from core.model_factory import build_chat_model, build_embeddings, load_model_config
from core.prompts import answer_prompt, intent_prompt, rewrite_prompt, route_prompt
from core.retrieval import build_bm25_index_for_vectorstore, set_rerank_model, set_vectorstore
from core.tools import ALL_TOOLS, MOCK_ORDERS, create_ticket_record, query_order_status

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")

RUN_DIR = Path(__file__).resolve().parent
RESULT_DIR = RUN_DIR / "results"
RESULT_DIR.mkdir(exist_ok=True)

INTENT_NAMES = {1: "知识库咨询", 2: "查订单", 3: "创建工单", 4: "日常闲聊"}

# 20 题金标集（覆盖全部题型：RAG/工具/闲聊/兜底/对抗/隐私/口语/情绪）
GOLD_IDS = ["02", "03", "13", "14", "19", "23", "27", "28", "29", "37",
            "39", "41", "44", "45", "47", "52", "55", "56", "61", "62"]

# 评测中视为"通用词"不计入事实覆盖（答案里必然出现、无区分度）
GENERIC_WORDS = {"问题", "客服", "订单", "商品", "可以", "我们", "您的", "请联", "联系",
                 "提供", "申请", "退货", "退款", "保修", "发票", "发货", "维修", "优惠券",
                 "工作日", "小时", "用户", "消费者", "政策", "说明", "相关", "信息", "处理"}


# ---------------------------------------------------------------
# 数据加载
# ---------------------------------------------------------------
def load_samples() -> list[dict]:
    rows = []
    with open(ROOT / "data" / "eval_set.csv", encoding="utf-8-sig") as f:
        for r in csv.DictReader(f):
            rows.append({
                "sid": r["序号"].strip(),
                "question": r["question"].strip(),
                "expected_intent": r["expected_intent"].strip(),
                "ground_truth": r["ground_truth"].strip(),
                "category": r.get("category", "").strip(),
                "source_doc": r.get("source_doc", "").strip(),
            })
    return rows


# ---------------------------------------------------------------
# 运行时构建（被测/裁判分离）
# ---------------------------------------------------------------
def build_components():
    cfg = load_model_config()
    sut = build_chat_model(cfg)  # 被测 deepseek-chat
    judge_cfg = json.loads(json.dumps(cfg))
    judge_cfg["chat"] = dict(judge_cfg.get("chat", {}))
    judge_cfg["chat"]["model"] = "deepseek-reasoner"
    judge = build_chat_model(judge_cfg)  # 裁判 deepseek-reasoner
    embeddings = build_embeddings(cfg)

    # 评测专用：API 请求超时（秒），防止偶发挂起拖死整批；在首次调用前赋值即生效
    for mdl in (sut, judge):
        try:
            mdl.request_timeout = 60
        except Exception:
            pass

    vs = Chroma(persist_directory=str(ROOT / "chroma_db"), embedding_function=embeddings)
    set_vectorstore(vs)
    bm25_data = build_bm25_index_for_vectorstore(vs)

    rewrite_chain = rewrite_prompt | sut | StrOutputParser()
    answer_chain = answer_prompt | sut | StrOutputParser()
    intent_chain = intent_prompt | sut | StrOutputParser()
    route_llm = sut.bind_tools(ALL_TOOLS)

    # 与生产完全一致的 Function Calling 路由（移植自 智能客服助手._make_intent_fns）
    def _make_route():
        def classify_intent(question, history=""):
            try:
                raw = intent_chain.invoke({"history": history, "question": question}).strip()
                m = re.search(r"^\s*([1-4])\s*$", raw) or re.search(r"([1-4])", raw)
                return int(m.group(1)) if m else 1
            except Exception:
                return 1

        def route_intent(question, history=""):
            try:
                ai = route_llm.invoke(
                    route_prompt.invoke({"history": history, "question": question}).to_messages()
                )
                tool_calls = getattr(ai, "tool_calls", None) or []
                if tool_calls:
                    tc = tool_calls[0]
                    intent = 2 if tc["name"] == "query_order" else 3
                    return {"intent": intent, "tool_args": dict(tc.get("args") or {})}
                content = (ai.content or "").strip().upper()
                return {"intent": 4 if "CHAT" in content else 1, "tool_args": {}}
            except Exception:
                return {"intent": classify_intent(question, history), "tool_args": {}}
        return route_intent

    def _retrieve(query, top_k, similarity_threshold, retrieval_mode, sources=None):
        return R.do_retrieve(query, top_k, similarity_threshold, retrieval_mode, sources,
                             vs=vs, bm25_data=bm25_data)

    def run_order_query(question, tool_args=None):
        """与生产 智能客服助手.run_order_query 完全一致"""
        tool_args = tool_args or {}
        # 隐私拦截：不允许用手机号查询订单
        if re.search(r"1[3-9]\d{9}", question) or re.fullmatch(
            r"1[3-9]\d{9}", str(tool_args.get("order_id") or "")
        ):
            return (
                "为了保护您的账户与隐私安全，暂不支持通过手机号查询订单。"
                "请在「我的订单」中找到订单号后发给我（演示环境可试用："
                f"{'、'.join(MOCK_ORDERS.keys())}），我马上为您查询。",
                "手机号查询→隐私拦截澄清",
            )
        order_id = str(tool_args.get("order_id") or "").strip()
        if not order_id:
            m = re.search(r"\d{6,8}", question)
            order_id = m.group() if m else ""
        if not order_id:
            return (
                "好的，我来帮您查询订单。请提供您的订单号"
                f"（演示环境可试用：{'、'.join(MOCK_ORDERS.keys())}）。",
                "参数不足（order_id 缺失），主动澄清",
            )
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

    def run_ticket_create(question, tool_args=None):
        """与生产 智能客服助手.run_ticket_create 完全一致"""
        tool_args = tool_args or {}
        issue = str(tool_args.get("issue") or "").strip() or question.strip()
        if len(issue) < 10:
            return (
                "好的，我来为您登记问题并创建工单。为了更快解决，请具体描述一下：\n\n"
                "1. 遇到了什么问题？\n2. 大概什么时间发生的？",
                "信息不足（issue 过短），主动澄清",
            )
        priority = str(tool_args.get("priority") or "").strip().upper()
        if priority not in ("P1", "P2"):
            priority = "P1" if re.search(r"投诉|紧急|马上|立刻|严重|法院|举报", question) else "P2"
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

    from langchain_core.prompts import ChatPromptTemplate
    # 与生产一致的闲聊链（移植自 智能客服助手.chitchat_prompt）
    chitchat_chain = (
        ChatPromptTemplate.from_messages([
            ("system", "你是智能客服助手'小智'，语气亲切、简洁。只做日常寒暄，不要编造任何公司政策或产品信息。"
                       "如果用户问到业务问题，友好地引导他直接提问，你会帮他查询知识库或办理业务。"),
            ("human", "{question}"),
        ])
        | sut | StrOutputParser()
    )

    graph = build_agent_graph({
        "route_intent": _make_route(),
        "run_order_query": run_order_query,
        "run_ticket_create": run_ticket_create,
        "rewrite_chain": rewrite_chain,
        "answer_chain": answer_chain,
        "chitchat_chain": chitchat_chain,
        "do_retrieve": _retrieve,
        "rerank_docs": R.rerank_docs,
        "dense_top_confidence": lambda q, sources=None: R.dense_top_confidence(q, sources, vs=vs),
        "format_docs": R.format_docs,
        "intent_names": INTENT_NAMES,
        "logger": __import__("logging").getLogger("batch_eval"),
    })
    return {"graph": graph, "sut": sut, "judge": judge, "embeddings": embeddings, "cfg": cfg}


# ---------------------------------------------------------------
# L0 硬指标
# ---------------------------------------------------------------
def fact_coverage(ground_truth: str, docs_text: str) -> float | None:
    """标准答案中的数字事实 + 2字以上关键词在召回文档中的覆盖率"""
    nums = set(re.findall(r"\d+(?:\.\d+)?%?", ground_truth))
    words = [w for w in jieba.cut(ground_truth) if len(w) >= 2 and w not in GENERIC_WORDS]
    total = len(nums) + len(words)
    if not total:
        return None
    hit = sum(1 for n in nums if n in docs_text) + sum(1 for w in words if w in docs_text)
    return round(hit / total, 3)


def hard_check(sample: dict, state: dict, latency: float) -> dict:
    exp = normalize_intent(sample["expected_intent"]) or sample["expected_intent"]
    intent_actual = state.get("intent_name", "")
    intent_ok = intent_actual == exp
    note = state.get("route_note", "") or ""
    docs = state.get("docs", []) or []
    docs_text = "\n".join(d.page_content for d in docs)
    answer = state.get("answer", "") or ""

    out = {
        "intent_ok": intent_ok, "intent_expected": exp, "intent_actual": intent_actual,
        "doc_count": len(docs), "fact_coverage": fact_coverage(sample["ground_truth"], docs_text),
        "fallback_triggered": "置信度兜底" in note,
        "tool_ok": None, "tool_detail": None, "reject_ok": None,
        "latency": round(latency, 2), "answer": answer, "route_note": note,
        "rewritten": state.get("rewritten_q", ""),
    }

    # 工具类硬判定
    if exp == "查订单":
        # 含手机号：期望隐私拦截澄清；有订单号期望查询成功；都没有期望主动澄清
        if re.search(r"1[3-9]\d{9}", sample["question"]):
            out["tool_ok"] = "隐私拦截" in note or "澄清" in note
        elif re.search(r"\d{6,8}", sample["question"]):
            out["tool_ok"] = "成功" in note
        else:
            out["tool_ok"] = "澄清" in note
        out["tool_detail"] = note
    elif exp == "创建工单":
        out["tool_ok"] = ("ticket" in str(state.get("trace", [])) or "create_ticket" in note) and "成功" in note
        out["tool_detail"] = note

    # 域外拒答题（source_doc=未命中兜底）：应婉拒/转人工且不编造具体政策
    if sample["source_doc"] == "未命中兜底":
        reject_words = ["无法", "抱歉", "暂不", "没有", "不在", "建议", "联系", "人工", "范围", "无法查询"]
        out["reject_ok"] = any(w in answer for w in reject_words) and len(answer) > 5
    return out


# ---------------------------------------------------------------
# L1 reasoner 裁判（锚点化 rubric）
# ---------------------------------------------------------------
JUDGE_PROMPT = """你是资深客服质检专家。对下面这条客服回答按三个维度各打 0-10 的整数分。

【评分锚点】
答案相关性（是否切题并真正解决问题）：
9-10 完整准确，覆盖全部关键信息；6-8 回应正确但略有遗漏；3-5 沾边但缺关键信息；0-2 答非所问或错误
忠实度（内容是否有据可依、无编造；对闲聊题为是否守边界/不泄露内部信息）：
9-10 全部可核实且无多余承诺；6-8 主体可靠有轻微泛化表述；3-5 含明显编造的金额/条款/承诺；0-2 大面积虚构或违规
检索相关性（召回资料是否包含答题所需核心信息；非知识库题型填 -1）：
9-10 含全部核心信息；6-8 含大部分；3-5 仅沾边；0-2 完全无关；非知识题填 -1

【题型】{itype}
【用户问题】{q}
【标准答案要点】{gt}
【召回资料摘要】{docs}
【客服回答】{ans}

只输出 JSON：{{"relevance":分,"faithfulness":分,"retrieval_relevance":分}}"""


def judge_answer(comp, sample, state) -> dict:
    answer = state.get("answer", "") or ""
    docs = state.get("docs", []) or []
    if not answer or "【" in answer[:3]:
        return {"relevance": None, "faithfulness": None, "retrieval_relevance": None, "judge_err": "无有效回答"}
    docs_brief = "\n".join(f"- {d.page_content[:180]}" for d in docs[:3]) or "（本题未走知识库检索）"
    itype = {"查订单": "工具调用", "创建工单": "工具调用", "日常闲聊": "闲聊/情绪"}.get(
        normalize_intent(sample["expected_intent"]) or "", "知识库问答")
    prompt = JUDGE_PROMPT.format(
        itype=itype, q=sample["question"], gt=sample["ground_truth"][:400],
        docs=docs_brief[:1200], ans=answer[:1500])
    for attempt in range(2):
        try:
            resp = comp["judge"].invoke(prompt)
            m = re.search(r"\{.*\}", resp.content, re.DOTALL)
            if m:
                d = json.loads(m.group())
                return {"relevance": float(d.get("relevance", 0)),
                        "faithfulness": float(d.get("faithfulness", 0)),
                        "retrieval_relevance": float(d.get("retrieval_relevance", -1)),
                        "judge_err": None}
        except Exception as e:
            err = str(e)
    return {"relevance": None, "faithfulness": None, "retrieval_relevance": None, "judge_err": err[:120]}


# ---------------------------------------------------------------
# 单题执行
# ---------------------------------------------------------------
def run_one(comp, sample, gcfg):
    set_rerank_model("bge-reranker-base（更快，CPU 友好）")
    state_in = {
        "question": sample["question"], "history": "",
        "top_k": gcfg["top_k"], "similarity_threshold": gcfg.get("threshold", 0.0),
        "retrieval_mode": gcfg["mode"],
        "rerank_enabled": gcfg["rerank"], "rerank_candidates": gcfg["rerank_k"],
        "confidence_fallback_enabled": gcfg["fallback_on"],
        "confidence_threshold": gcfg["fallback_th"],
        "active_sources": None,
    }
    t0 = time.time()
    try:
        state = comp["graph"].invoke(state_in)
        latency = time.time() - t0
        err = None
    except Exception as e:
        state, latency, err = {}, time.time() - t0, str(e)[:200]
    rec = {"sid": sample["sid"], "config": gcfg["name"], "error": err}
    if err:
        rec.update({"hard": None, "judge": None})
        return rec
    rec["hard"] = hard_check(sample, state, latency)
    rec["judge"] = judge_answer(comp, sample, state)
    return rec


def run_group(comp, samples, gcfg, workers: int = 6) -> list[dict]:
    """并发跑一组配置，断点续跑：已存在的结果文件直接跳过"""
    path = RESULT_DIR / f"{gcfg['name']}.json"
    done = {}
    if path.exists():
        try:
            # 带 error 的记录（如 402 余额不足）视为未完成，续跑时自动重试
            done = {r["sid"]: r for r in json.loads(path).get("records", []) if not r.get("error")}
        except Exception:
            done = {}
    todo = [s for s in samples if s["sid"] not in done]
    recs = list(done.values())
    print(f"[{gcfg['name']}] 共 {len(samples)} 题，已完成 {len(done)}，待跑 {len(todo)}")
    for i in range(0, len(todo), workers):
        batch = todo[i:i + workers]
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(run_one, comp, s, gcfg): s for s in batch}
            for fu in as_completed(futs):
                recs.append(fu.result())
        recs.sort(key=lambda r: r["sid"])
        path.write_text(json.dumps({"config": gcfg, "records": recs}, ensure_ascii=False, indent=1),
                        encoding="utf-8")
        n = len(recs)
        errs = sum(1 for r in recs if r.get("error"))
        print(f"  进度 {n}/{len(samples)}（异常 {errs}）", flush=True)
    return recs
