# =========================================================
# 模型设置页
# =========================================================
import streamlit as st

from 智能客服助手 import get_runtime, _build_runtime, logger, LOG_DIR
from core.model_factory import (
    LOCAL_CHAT_MODEL, LOCAL_EMBEDDING_MODEL, OLLAMA_BASE_URL,
    CHAT_PRESETS, EMBEDDING_PRESETS,
    load_model_config, save_model_config, update_kb_space,
    build_chat_model, build_embeddings,
    embedding_space, kb_compatibility, describe_source,
    resolve_api_key, set_session_api_key, api_key_source,
    test_chat_connection, test_embedding_connection,
    upsert_model_profile, delete_model_profile, list_local_ollama_models,
    list_local_chat_models,
    list_local_embedding_models,
)
from core.retrieval import rebuild_knowledge_base, RERANK_MODEL_OPTIONS

st.set_page_config(page_title="模型设置", page_icon="🧠", layout="wide")
st.title("🧠 模型设置")


def _render_api_key_status(provider: str, input_state_key: str) -> None:
    """展示 Key 状态：输入框有新值（未生效） vs 已生效 Key 的来源（会话 / .env 全局）。"""
    _typed = (st.session_state.get(input_state_key) or "").strip()
    if _typed:
        st.caption("⚠️ 输入框中的新 Key **尚未生效**：点「测试连接」即用它验证，点「保存并生效」后本次会话内使用（不落盘）。")
        return
    _saved = resolve_api_key(provider)
    if not _saved:
        st.caption("尚未配置 Key：粘贴你的 Key 后点「保存并生效」（仅本次会话使用，不落盘；长期使用请配置到 .env）。")
        return
    _src = api_key_source(provider)
    if _src == "session":
        _label = "本次会话临时输入"
    else:
        _label = ".env 环境变量"
    st.caption(f"🔑 当前生效 Key 来源：{_label}（{_saved[:6]}****）。输入框留空即沿用它。")

_RT = get_runtime()
_cfg = load_model_config()
_kb_ok, _cur_space, _kb_space = kb_compatibility(_RT["cfg"])

_save_msg = st.session_state.pop("_save_msg", None)
if _save_msg == "ok":
    st.success("✅ 配置已保存并生效")
elif _save_msg == "warn":
    st.warning("⚠️ 向量空间已变更，请在下方「重建知识库」完成迁移。")
if _RT.get("warning"):
    st.info(f"ℹ️ {_RT['warning']}")

# =========================================================
# ❶ 聊天模型 + 向量模型（左右两列，带边框容器）
# =========================================================
_col_chat, _col_emb = st.columns(2)

with _col_chat:
    with st.container(border=True):
        st.subheader("💬 聊天模型")
        _chat = _cfg["chat"]
        _chat_src_label = st.radio(
            "来源", ["本地 Ollama", "API"],
            index=0 if _chat["source"] == "local" else 1,
            horizontal=True, key="cfg_chat_source",
        )
        _new_chat = {"source": "local" if _chat_src_label.startswith("本地") else "api"}
        if _new_chat["source"] == "local":
            _local_chat = list_local_chat_models()
            _chat_default = _chat.get("model") or LOCAL_CHAT_MODEL
            _chat_opts = _local_chat or [_chat_default]
            _sel = st.selectbox(
                "模型", _chat_opts,
                index=_chat_opts.index(_chat_default) if _chat_default in _chat_opts else 0,
                key="cfg_local_chat_model",
            )
            _new_chat.update({"provider": _chat.get("provider", "自定义"), "base_url": "", "model": _sel})
        else:
            _c_providers = list(CHAT_PRESETS)
            _c_prov = st.selectbox(
                "服务商", _c_providers,
                index=_c_providers.index(_chat["provider"]) if _chat["provider"] in _c_providers else 0,
                key="cfg_chat_provider",
            )
            _cpreset = CHAT_PRESETS[_c_prov]
            _new_chat["provider"] = _c_prov
            _new_chat["base_url"] = _chat["base_url"] if _chat["provider"] == _c_prov else _cpreset["base_url"]
            _chat_base = st.text_input(
                "接口地址 (Base URL)",
                value=_new_chat["base_url"] or _cpreset["base_url"],
                key=f"cfg_chat_baseurl_{_c_prov}",
                help="OpenAI 兼容接口地址。使用中转/代理服务时填其地址；填回官方地址即等效于留空。本地 Ollama 无需此项。",
            )
            _new_chat["base_url"] = _chat_base.strip()
            if _cpreset["models"]:
                _c_opts = _cpreset["models"] + ["自定义模型名..."]
                _c_idx = (_c_opts.index(_chat["model"])
                          if _chat["provider"] == _c_prov and _chat["model"] in _cpreset["models"] else 0)
                _c_sel = st.selectbox("模型", _c_opts, index=_c_idx, key=f"cfg_chat_model_{_c_prov}")
                _new_chat["model"] = (st.text_input("模型名称", value=_chat.get("model", ""), key=f"cfg_chat_custom_{_c_prov}")
                                      if _c_sel == "自定义模型名..." else _c_sel)
            else:
                _new_chat["model"] = st.text_input("模型名称", value=_chat.get("model", ""), key=f"cfg_chat_custom_{_c_prov}")
            _c_input_key = f"cfg_chat_key_{_c_prov}"
            st.text_input(
                "API Key",
                type="password",
                key=_c_input_key,
                help="粘贴你的 Key 后点「测试连接」验证、点「保存并生效」本次会话内生效（不落盘）；长期使用请在 .env 配置。留空则沿用当前生效的 Key。",
            )
            _render_api_key_status(_c_prov, _c_input_key)

with _col_emb:
    with st.container(border=True):
        st.subheader("🧲 向量模型")
        _emb = _cfg["embedding"]
        _emb_src_label = st.radio(
            "来源", ["本地 Ollama", "API"],
            index=0 if _emb["source"] == "local" else 1,
            horizontal=True, key="cfg_emb_source",
        )
        _new_emb = {"source": "local" if _emb_src_label.startswith("本地") else "api"}
        if _new_emb["source"] == "local":
            _local_emb = list_local_embedding_models()
            _emb_default = _emb.get("model") or LOCAL_EMBEDDING_MODEL
            _emb_opts = _local_emb or [_emb_default]
            _sel = st.selectbox(
                "模型", _emb_opts,
                index=_emb_opts.index(_emb_default) if _emb_default in _emb_opts else 0,
                key="cfg_local_emb_model",
            )
            _new_emb.update({"provider": _emb.get("provider", "自定义"), "base_url": "", "model": _sel})
            if not _kb_ok:
                st.caption(f"⚠️ 空间不一致：`{_cur_space}` ≠ 建库 `{_kb_space}`")
        else:
            _e_providers = list(EMBEDDING_PRESETS)
            _e_prov = st.selectbox(
                "服务商", _e_providers,
                index=_e_providers.index(_emb["provider"]) if _emb["provider"] in _e_providers else 0,
                key="cfg_emb_provider",
            )
            _epreset = EMBEDDING_PRESETS[_e_prov]
            _new_emb["provider"] = _e_prov
            _new_emb["base_url"] = _emb["base_url"] if _emb["provider"] == _e_prov else _epreset["base_url"]
            _emb_base = st.text_input(
                "接口地址 (Base URL)",
                value=_new_emb["base_url"] or _epreset["base_url"],
                key=f"cfg_emb_baseurl_{_e_prov}",
                help="OpenAI 兼容 embeddings 接口地址。使用中转/代理服务时填其地址；填回官方地址即等效于留空。",
            )
            _new_emb["base_url"] = _emb_base.strip()
            if _epreset["models"]:
                _e_opts = _epreset["models"] + ["自定义模型名..."]
                _e_idx = (_e_opts.index(_emb["model"])
                          if _emb["provider"] == _e_prov and _emb["model"] in _epreset["models"] else 0)
                _e_sel = st.selectbox("模型", _e_opts, index=_e_idx, key=f"cfg_emb_model_{_e_prov}")
                _new_emb["model"] = (st.text_input("模型名称", value=_emb.get("model", ""), key=f"cfg_emb_custom_{_e_prov}")
                                     if _e_sel == "自定义模型名..." else _e_sel)
            else:
                _new_emb["model"] = st.text_input("模型名称", value=_emb.get("model", ""), key=f"cfg_emb_custom_{_e_prov}")
            _e_input_key = f"cfg_emb_key_{_e_prov}"
            st.text_input(
                "API Key",
                type="password",
                key=_e_input_key,
                help="粘贴你的 Key 后点「测试连接」验证、点「保存并生效」本次会话内生效（不落盘）；长期使用请在 .env 配置。留空则沿用当前生效的 Key。",
            )
            _render_api_key_status(_e_prov, _e_input_key)
            st.caption("⚠️ 不同向量空间切换后需重建知识库")

# =========================================================
# ❷ 操作按钮
# =========================================================
_b1, _b2 = st.columns(2)
_do_test = _b1.button("🔌 测试连接", use_container_width=True, key="cfg_test_btn")
_do_save = _b2.button("💾 保存并生效", type="primary", use_container_width=True, key="cfg_save_btn")

if _do_test:
    if _new_chat.get("source") == "api":
        _tk = st.session_state.get(f"cfg_chat_key_{_new_chat.get('provider', '')}")
        if _tk:
            set_session_api_key(_new_chat["provider"], _tk)
    if _new_emb.get("source") == "api":
        _ek = st.session_state.get(f"cfg_emb_key_{_new_emb.get('provider', '')}")
        if _ek:
            set_session_api_key(_new_emb["provider"], _ek)
    with st.spinner("测试聊天模型..."):
        _okc, _msgc = test_chat_connection({**_cfg, "chat": _new_chat})
    (st.success if _okc else st.error)(f"💬 {_msgc}")
    with st.spinner("测试向量模型..."):
        _oke, _msge = test_embedding_connection({**_cfg, "embedding": _new_emb})
    (st.success if _oke else st.error)(f"🧲 {_msge}")

if _do_save:
    if _new_chat.get("source") == "api":
        _tk = st.session_state.get(f"cfg_chat_key_{_new_chat.get('provider', '')}")
        if _tk:
            set_session_api_key(_new_chat["provider"], _tk)
    if _new_emb.get("source") == "api":
        _ek = st.session_state.get(f"cfg_emb_key_{_new_emb.get('provider', '')}")
        if _ek:
            set_session_api_key(_new_emb["provider"], _ek)
    _new_cfg = {**_cfg, "chat": _new_chat, "embedding": _new_emb}
    try:
        if _new_chat["source"] == "api":
            build_chat_model(_new_cfg)
        if _new_emb["source"] == "api":
            build_embeddings(_new_cfg)
    except Exception as e:
        st.error(f"配置无法生效：{e}")
        st.stop()
    _space_changed = embedding_space(_new_cfg) != _cfg.get("kb_embedding_space", LOCAL_EMBEDDING_MODEL)
    save_model_config(_new_cfg)
    _build_runtime.clear()
    st.session_state["_need_rebuild"] = _space_changed
    st.session_state["_save_msg"] = "warn" if _space_changed else "ok"
    logger.info(
        f"模型配置已保存 | chat={_new_chat.get('source')}/{_new_chat.get('model', '')} "
        f"emb={_new_emb.get('source')}/{_new_emb.get('model', '')} 空间变更={_space_changed}"
    )
    st.rerun()

# =========================================================
# ❸ 重建知识库（仅在空间不一致时显示）
# =========================================================
_need_rebuild = st.session_state.get("_need_rebuild", False)
if not _kb_ok or _need_rebuild:
    with st.container(border=True):
        st.subheader("🔄 重建知识库")
        st.error(f"⚠️ 向量空间不一致：`{_cur_space}` ≠ 建库 `{_kb_space}`，检索已暂停。")
        if st.button("🔄 重建知识库", type="primary", use_container_width=True, key="kb_rebuild_btn"):
            _progress = st.progress(0.0, text="准备重建...")

            def _rebuild_cb(done: int, total: int, name: str):
                _progress.progress(done / max(total, 1), text=f"重建中 {done + 1}/{total}：{name}")

            try:
                with st.spinner("正在重建知识库..."):
                    _result = rebuild_knowledge_base(_RT["vectorstore"], progress_cb=_rebuild_cb)
                update_kb_space(_cur_space)
                st.session_state["_need_rebuild"] = False
                _progress.progress(1.0, text="重建完成")
                if _result["failed"]:
                    st.warning("以下文件重建失败：" + "；".join(f"{n}（{r}）" for n, r in _result["failed"]))
                st.success(f"✅ 重建完成：{_result['files']} 个文件，{_result['chunks']} 个向量块")
                logger.info(f"知识库重建完成 | files={_result['files']} chunks={_result['chunks']}")
            except Exception as e:
                st.error(f"重建失败：{e}")
                logger.error(f"知识库重建失败：{e}")

# =========================================================
# ❹ 模型库（折叠）
# =========================================================
with st.expander("📚 模型库", expanded=False):
    _chat_tab, _emb_tab = st.tabs(["💬 聊天模型", "🧲 向量模型"])

    with _chat_tab:
        _r1, _r2 = st.columns([3, 1])
        with _r1:
            _chat_pn = st.text_input("名称", value=_new_chat.get("model") or "当前聊天模型", key="chat_profile_name", label_visibility="collapsed")
        with _r2:
            if st.button("💾 保存当前", use_container_width=True, key="save_chat_profile"):
                _lib_cfg = load_model_config()
                upsert_model_profile(_lib_cfg, "chat", _chat_pn, _new_chat)
                save_model_config(_lib_cfg)
                _build_runtime.clear()
                st.success("✅ 已保存")
                st.rerun()

        _chat_profiles = (_cfg.get("model_profiles") or {}).get("chat", [])
        if _chat_profiles:
            st.markdown("---")
            for _p in _chat_profiles:
                _pid = _p.get("id", "")
                _ca, _cb, _cc = st.columns([6, 1, 1])
                with _ca:
                    st.markdown(f"**{_p.get('name', '未命名')}** 　`{_p.get('model', '')}`")
                    st.caption(f"{_p.get('source', '')} · {_p.get('base_url') or '本地'}")
                with _cb:
                    if st.button("启用", key=f"act_chat_{_pid}", use_container_width=True):
                        _ac = load_model_config()
                        _ac["chat"] = {k: _p.get(k, "") for k in ("source", "provider", "base_url", "model")}
                        save_model_config(_ac)
                        _build_runtime.clear()
                        st.session_state["_save_msg"] = "ok"
                        st.rerun()
                with _cc:
                    if st.button("🗑️", key=f"del_chat_{_pid}"):
                        _lib_cfg = load_model_config()
                        delete_model_profile(_lib_cfg, "chat", _pid)
                        save_model_config(_lib_cfg)
                        st.rerun()
        else:
            st.caption("暂无已保存的聊天模型配置。")

    with _emb_tab:
        _r1, _r2 = st.columns([3, 1])
        with _r1:
            _emb_pn = st.text_input("名称", value=_new_emb.get("model") or LOCAL_EMBEDDING_MODEL, key="emb_profile_name", label_visibility="collapsed")
        with _r2:
            if st.button("💾 保存当前", use_container_width=True, key="save_emb_profile"):
                _lib_cfg = load_model_config()
                upsert_model_profile(_lib_cfg, "embedding", _emb_pn, _new_emb)
                save_model_config(_lib_cfg)
                _build_runtime.clear()
                st.success("✅ 已保存")
                st.rerun()

        _emb_profiles = (_cfg.get("model_profiles") or {}).get("embedding", [])
        if _emb_profiles:
            st.markdown("---")
            for _p in _emb_profiles:
                _pid = _p.get("id", "")
                _ca, _cb, _cc = st.columns([6, 1, 1])
                with _ca:
                    st.markdown(f"**{_p.get('name', '未命名')}** 　`{_p.get('model', '')}`")
                    st.caption(f"{_p.get('source', '')} · {_p.get('base_url') or '本地'}")
                with _cb:
                    if st.button("启用", key=f"act_emb_{_pid}", use_container_width=True):
                        _ac = load_model_config()
                        _ac["embedding"] = {k: _p.get(k, "") for k in ("source", "provider", "base_url", "model")}
                        _sc = embedding_space(_ac) != _ac.get("kb_embedding_space", LOCAL_EMBEDDING_MODEL)
                        save_model_config(_ac)
                        _build_runtime.clear()
                        st.session_state["_need_rebuild"] = _sc
                        st.session_state["_save_msg"] = "warn" if _sc else "ok"
                        st.rerun()
                with _cc:
                    if st.button("🗑️", key=f"del_emb_{_pid}"):
                        _lib_cfg = load_model_config()
                        delete_model_profile(_lib_cfg, "embedding", _pid)
                        save_model_config(_lib_cfg)
                        st.rerun()
        else:
            st.caption("暂无已保存的向量模型配置。")


# =========================================================
# ❺ 问答策略（全局默认值：检索 / Rerank / 置信度兜底 / 上下文压缩）
#    与主页问答、评测页共用同一组 session_state 参数，修改后下一次提问即生效；
#    评测页「应用到生产」写入的也是这组参数。对话页保持纯产品展示，不暴露参数。
# =========================================================
# 兜底初始化：不经过对话页直接打开本页时，session_state 还没有这些键（主页 defaults 才会创建），
# 这里用与主页 defaults 相同的初始值补齐，避免 AttributeError。
_QA_DEFAULTS = {
    "retrieval_mode": "向量检索",
    "top_k": 5,
    "similarity_threshold": 0.0,
    "rerank_enabled": True,  # 与置信度兜底联动（兜底依赖精排分数）
    "rerank_model": "bge-reranker-base（更快，CPU 友好）",
    "rerank_candidates": 10,
    "confidence_fallback_enabled": True,
    "confidence_threshold": 0.4,
    "history_compress_enabled": True,
    "history_compress_threshold": 2000,
}
for _qk, _qv in _QA_DEFAULTS.items():
    st.session_state.setdefault(_qk, _qv)

with st.expander("🔍 问答策略（全局默认值）", expanded=False):
    st.caption("对话页只展示回答效果，问答参数统一在这里配置；修改后对下一次提问生效，无需重启。")

    st.markdown("##### 🔍 检索策略")
    mode_options = ["向量检索", "关键词", "混合检索"]
    st.session_state.retrieval_mode = st.radio(
        "默认检索方式",
        options=mode_options,
        index=mode_options.index(st.session_state.retrieval_mode)
        if st.session_state.retrieval_mode in mode_options
        else 0,
        horizontal=True,
        help=(
            "向量检索：基于语义相似度（bge-m3），适合同义改写、模糊描述的问题；\n"
            "关键词：基于 BM25 词项匹配（jieba 分词），适合精确查找专有名词、编号；\n"
            "混合检索：两者结合并用 RRF 融合排序，兼顾语义与精确匹配。"
        ),
    )
    st.session_state.top_k = st.slider(
        "检索数量（TOP_K）",
        min_value=1,
        max_value=10,
        value=st.session_state.top_k,
        step=1,
        help="精排/召回后最终返回给大模型的参考文档块数量。评测结论：5 为最优。",
    )
    st.session_state.similarity_threshold = st.slider(
        "相似度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.similarity_threshold,
        step=0.05,
        help="向量检索时，余弦相似度低于此值的文档会被过滤。0 = 不过滤。建议 0.3~0.5。",
    )

    st.divider()
    st.markdown("##### 🎯 Rerank 精排")
    st.session_state.rerank_enabled = st.checkbox(
        "启用 Rerank 重排序",
        value=st.session_state.rerank_enabled,
        help=(
            "开启后先粗检索更多候选块，再用交叉编码器按相关性精排，"
            "可提升检索质量。首次使用需下载模型（约 1GB），CPU 推理稍慢。"
        ),
    )
    st.session_state.rerank_model = st.selectbox(
        "Rerank 模型",
        options=list(RERANK_MODEL_OPTIONS.keys()),
        index=list(RERANK_MODEL_OPTIONS.keys()).index(st.session_state.rerank_model)
        if st.session_state.rerank_model in RERANK_MODEL_OPTIONS
        else 0,
        disabled=not st.session_state.rerank_enabled,
    )
    st.session_state.rerank_candidates = st.slider(
        "粗排候选数量",
        min_value=5,
        max_value=30,
        value=st.session_state.rerank_candidates,
        step=1,
        help="先检索多少个候选块送入 Rerank 模型精排，最终仍返回 TOP_K 个。建议为 TOP_K 的 4~6 倍。",
        disabled=not st.session_state.rerank_enabled,
    )
    if st.session_state.rerank_enabled:
        st.caption(f"✅ 已启用：{RERANK_MODEL_OPTIONS[st.session_state.rerank_model]}")
    else:
        st.caption("未启用（仅使用粗排结果）")

    st.divider()
    st.markdown("##### 🛡️ 置信度兜底")
    st.session_state.confidence_fallback_enabled = st.checkbox(
        "启用置信度兜底",
        value=st.session_state.confidence_fallback_enabled,
        help=(
            "生产级客服策略：精排最高相关性得分低于阈值时，不强行作答，"
            "回复转人工提示，避免模型编造答案。"
        ),
        disabled=not st.session_state.rerank_enabled,
    )
    st.session_state.confidence_threshold = st.slider(
        "置信度阈值",
        min_value=0.0,
        max_value=1.0,
        value=st.session_state.confidence_threshold,
        step=0.05,
        help="精排后最高分低于该值即触发兜底。建议 0.3~0.6，过高会导致频繁兜底。",
        disabled=not (st.session_state.rerank_enabled and st.session_state.confidence_fallback_enabled),
    )
    if st.session_state.rerank_enabled and st.session_state.confidence_fallback_enabled:
        st.caption(f"兜底策略：最高相关性 < {st.session_state.confidence_threshold:.0%} 时回复转人工提示")
    elif st.session_state.rerank_enabled:
        st.caption("兜底未启用（无论得分高低都会作答）")
    else:
        st.caption("置信度兜底依赖 Rerank 精排，请先启用 Rerank")

    st.divider()
    st.markdown("##### 🧠 多轮上下文压缩")
    st.session_state.history_compress_enabled = st.checkbox(
        "启用上下文压缩",
        value=st.session_state.history_compress_enabled,
        help=(
            "对话轮数较多（>10 轮）时启用。自动把早期对话用 LLM 总结成几条要点，"
            "保留近期原文，避免老消息被截断丢弃导致意图失忆。"
        ),
    )
    st.session_state.history_compress_threshold = st.slider(
        "触发压缩的 token 阈值",
        min_value=800,
        max_value=4000,
        value=st.session_state.history_compress_threshold,
        step=200,
        help="历史总 token 超过此值才触发压缩；低于此值保持原行为（按 token 截断）。",
        disabled=not st.session_state.history_compress_enabled,
    )
    if st.session_state.history_compress_enabled:
        st.caption(
            f"✅ 已启用：历史 ≥ {st.session_state.history_compress_threshold} tokens 时，"
            "老消息自动总结 + 近期原文拼接"
        )
    else:
        st.caption("未启用（仅按 token 截断最近消息，老消息会丢失）")


# =========================================================
# ❼ 运行日志（折叠面板，运维排查用）
#    原独立页 pages/05_运行日志.py 的逻辑内联在此，避免侧边栏多一个
#    几乎用不到的入口；展开后仍可按级别过滤、倒序查看最近 N 行。
# =========================================================
import re

def _read_recent_logs(n: int) -> list:
    log_file = LOG_DIR / "app.log"
    if not log_file.exists():
        return []
    try:
        with open(log_file, "r", encoding="utf-8") as f:
            return f.readlines()[-n:][::-1]
    except Exception:
        return []


with st.expander("🛠 运行日志（排查问题时展开）", expanded=False):
    st.caption(
        "按级别过滤、倒序展示最近 N 行。日志同时按 5MB 轮转写入 logs/app.log，"
        "生产环境排查问题直接看这里即可。"
    )
    _c1, _c2, _c3 = st.columns([1, 1, 1.2])
    with _c1:
        _log_level = st.selectbox(
            "级别过滤", ["全部", "ERROR", "WARNING", "INFO"], key="md_log_level_sel"
        )
    with _c2:
        _log_count = st.selectbox(
            "显示行数", [50, 100, 200, 500, 1000], index=1, key="md_log_count_sel"
        )
    with _c3:
        st.write("")
        if st.button("🔄 刷新日志", use_container_width=True, key="md_log_refresh_btn"):
            st.toast("已重新读取日志", icon="✅")

    _logs = _read_recent_logs(_log_count)
    if not _logs:
        st.info("暂无运行日志。产生对话、配置变更或知识库操作后会自动记录。")
    else:
        _filtered = []
        for _ln in _logs:
            _m = re.search(r"\[(ERROR|WARNING|INFO|DEBUG)\]", _ln)
            _lv = _m.group(1) if _m else "INFO"
            if _log_level == "全部" or _lv == _log_level:
                _filtered.append(_ln)

        if not _filtered:
            st.info(f"最近 {_log_count} 行中无 {_log_level} 级别记录。")
        else:
            _err = sum(1 for ln in _filtered if "[ERROR]" in ln)
            _warn = sum(1 for ln in _filtered if "[WARNING]" in ln)
            _info = sum(1 for ln in _filtered if "[INFO]" in ln)
            _stat_cols = st.columns(4)
            _stat_cols[0].metric("当前显示", f"{len(_filtered)} 条")
            _stat_cols[1].metric("ERROR", _err)
            _stat_cols[2].metric("WARNING", _warn)
            _stat_cols[3].metric("INFO", _info)
            st.caption(f"共 {len(_filtered)} 条（最新在前，共读取最近 {_log_count} 行）")
            st.code("".join(_filtered), language="text")
