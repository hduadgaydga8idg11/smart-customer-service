# =========================================================
# 知识库管理页面（独立页）
#   Tab1 上传与切分预览：上传 → 提取 → 切分 → 预览每块效果
#        → 可调参数重新预览 → 确认后才向量化入库（未确认不写库）
#   Tab2 已入库文档：查看切块详情、删除文件（二次确认）
# =========================================================
import streamlit as st
from pathlib import Path

from 智能客服助手 import get_runtime, logger
from core.retrieval import (
    MAX_FILE_SIZE_MB,
    calculate_file_hash,
    save_uploaded_file,
    is_file_already_in_db,
    split_file_for_preview,
    add_prepared_chunks_to_vectorstore,
    get_all_file_records,
    get_chunks_for_file,
    delete_file_from_vectorstore,
    build_bm25_index,
    bump_kb_version,
    SPLIT_STRATEGIES,
    read_raw_for_preview,
    DOCLING_ENABLED,
)
from core.model_factory import kb_compatibility

st.set_page_config(page_title="知识库管理", page_icon="📚", layout="wide")
st.title("📚 知识库管理")
st.caption("上传文档 → 预览切分效果 → 确认后向量化入库；入库文档自动参与对话页检索")

# 运行时资源：每次脚本执行取当前生效版本（主页切换模型后自动跟随）
_RT = get_runtime()
vectorstore = _RT["vectorstore"]

# 嵌入空间守卫：与知识库建库空间不一致时禁止入库，避免污染向量库
_KB_OK, _CUR_SPACE, _KB_SPACE = kb_compatibility(_RT["cfg"])
if not _KB_OK:
    st.error(
        f"⚠️ 上传入库已暂停：当前嵌入向量空间（{_CUR_SPACE}）与知识库建库空间"
        f"（{_KB_SPACE}）不一致。请先到左侧导航「🧠 模型设置」页完成迁移。"
    )
    st.page_link("pages/04_模型设置.py", label="🔄 前往模型设置页重建知识库")

# 上传格式：Docling 开启时支持全格式（旧版 .ppt 明确拒绝，请另存为 .pptx）；关闭时仅 TXT/Markdown
SUPPORTED_TYPES = (
    ["txt", "pdf", "docx", "xlsx", "xls", "csv", "html", "md", "pptx"]
    if DOCLING_ENABLED
    else ["txt", "md"]
)
_UPLOAD_HELP = (
    "支持：TXT、PDF、Word、Excel、CSV、HTML、Markdown、PowerPoint（.pptx；旧版 .ppt 请另存为 .pptx）"
    if DOCLING_ENABLED
    else "支持：TXT、Markdown（PDF/Word/图片等格式已停用，可在 .env 设 ENABLE_DOCLING=true 恢复）"
)
MAX_PREVIEW_CHUNKS = 50  # 预览最多展开显示的块数


def _clear_pending(delete_file: bool = True):
    """清空待入库状态（取消时删除已落盘的临时文件）"""
    pending = st.session_state.get("kb_pending")
    if pending and delete_file:
        p = Path(pending["path"])
        if p.exists():
            p.unlink()
    st.session_state.kb_pending = None


def _do_confirm_ingest(pending: dict, kb_ok: bool):
    """执行入库动作（顶部入库按钮复用）。"""
    if not kb_ok:
        st.warning("向量空间不一致，已暂停入库。请先到「🧠 模型设置」页完成迁移。")
        return
    with st.spinner("正在生成向量并写入知识库..."):
        try:
            success, msg = add_prepared_chunks_to_vectorstore(
                chunks=pending["chunks"],
                file_path=Path(pending["path"]),
                original_file_name=pending["name"],
                file_hash=pending["hash"],
            )
        except Exception as e:
            logger.error(f"入库异常 name={pending['name']}: {e}")
            success, msg = False, "入库失败，请查看服务日志。"
    if success:
        # 知识库变更：重建 BM25 索引；新文档默认关联勾选
        bump_kb_version()
        build_bm25_index.clear()
        selected = st.session_state.get("kb_selected")
        if selected is not None:
            selected.add(pending["name"])
        st.session_state.kb_pending = None
        st.success(msg + " 已自动关联到对话页。")
        st.balloons()
        st.rerun()
    else:
        st.warning(msg)


def _escape_md(text: str) -> str:
    """HTML 转义文档内容（用户上传，不可信），防 XSS；保留原有换行由调用方处理。"""
    import html as _html
    return _html.escape(str(text or ""), quote=False)


def _render_table_row_card(block_idx: int, chunk):
    """表格行切分的横排卡片：知识标题 / 相似问法 / 答案内容 三列并排展示。

    优先使用 metadata 中的结构化字段（title/questions/answer）；
    若超长被二次切分（只有部分字段为空），则降级显示 page_content。
    """
    title = chunk.metadata.get("title", "")
    questions = chunk.metadata.get("questions", "")
    answer = chunk.metadata.get("answer", "")
    others = chunk.metadata.get("others") or []
    if not (title or questions or answer):
        st.text(chunk.page_content)
        return
    c1, c2, c3 = st.columns([1, 1.2, 2], gap="small")
    with c1:
        st.caption("📌 知识标题")
        # 文档内容来自用户上传，一律 HTML 转义后渲染，防止表格内容注入网页脚本（XSS）
        st.markdown(_escape_md(title) or "—")
    with c2:
        st.caption("❓ 相似问法")
        st.markdown(_escape_md(questions).replace("\n", "<br>") if questions else "—", unsafe_allow_html=True)
    with c3:
        st.caption("💡 答案内容")
        st.markdown(_escape_md(answer) or "—")
    if others:
        with st.expander("附加信息", expanded=False):
            for line in others:
                st.markdown(f"- {_escape_md(line)}")


tab_upload, tab_files = st.tabs(["📤 上传与切分预览", "📚 已入库文档"])

# =========================================================
# Tab 1：上传 → 切分预览 → 确认入库
# =========================================================
with tab_upload:
    uploaded_file = st.file_uploader(
        "上传文档",
        type=SUPPORTED_TYPES,
        key="kb_uploader",
        help=_UPLOAD_HELP,
    )

    st.subheader("切分参数", divider="gray")

    # 切分方式选择
    strategy = st.selectbox(
        "切分方式",
        SPLIT_STRATEGIES,
        index=SPLIT_STRATEGIES.index(st.session_state.get("kb_split_strategy", "QA对切分"))
        if st.session_state.get("kb_split_strategy", "QA对切分") in SPLIT_STRATEGIES
        else 0,
        key="kb_split_strategy",
        help=(
            "**QA对切分**：按【问题】标记切分，一个问答对一块，最适合 Markdown 客服 FAQ\n\n"
            "**标题切分**：按 #/##/### 标题分节，适合有清晰目录的文档\n\n"
            "**语义切分**：按句子相似度在语义转折处切分，适合长文档/无结构文本\n\n"
            "**表格行切分**：按表格行切分，每行 = 一个块（知识标题/相似问法/答案内容），"
            "适合 xlsx/xls/csv 结构化 FAQ 表（仅 xlsx/xls/csv 生效）"
        ),
    )

    # 切块大小（三种方式均只需块大小，无需重合参数）
    _size_hint = {
        "QA对切分": "问答对超过此大小会二次切分（兜底阈值）",
        "标题切分": "章节超过此大小会二次切分（兜底阈值）",
        "语义切分": "目标块大小（语义切分会尽量凑近此值）",
        "表格行切分": "单行内容超过此大小会二次切分（兜底阈值）",
    }
    chunk_size = st.slider(
        "切块大小（字符数）", 100, 2000,
        st.session_state.get("kb_chunk_size", 512), step=10,
        help=_size_hint.get(strategy, ""),
    )
    chunk_overlap = 0  # 切分方式均无需重合参数
    st.session_state.kb_chunk_size = chunk_size
    st.session_state.kb_chunk_overlap = chunk_overlap

    # 切换了上传文件 → 清理上一个待入库状态
    if uploaded_file is not None:
        uploaded_bytes = uploaded_file.getvalue()
        current_hash = calculate_file_hash(uploaded_bytes)
        pending = st.session_state.get("kb_pending")
        if pending and pending["hash"] != current_hash:
            _clear_pending()

    st.divider()

    if uploaded_file is None:
        st.info("👆 请先选择要上传的文档")
    else:
        st.info(f"当前文件：**{uploaded_file.name}**")
        size_mb = len(uploaded_bytes) / 1024 / 1024

        if len(uploaded_bytes) > MAX_FILE_SIZE_MB * 1024 * 1024:
            st.error(
                f"文件大小 {size_mb:.1f}MB，超过 {MAX_FILE_SIZE_MB}MB 限制，请压缩或拆分后再上传。"
            )
        elif is_file_already_in_db(current_hash):
            st.warning("该文件内容已经在知识库中，不能重复入库。")
        else:
            pending = st.session_state.get("kb_pending")

            col_gen, col_cancel = st.columns([1, 1])
            gen_label = "🔄 重新生成切分预览" if pending else "🔍 生成切分预览"
            if col_gen.button(gen_label, type="primary", use_container_width=True):
                with st.spinner("正在读取文档并切分（PDF/图片含 OCR，可能需要几十秒）..."):
                    try:
                        saved_path = save_uploaded_file(uploaded_file, current_hash)
                        chunks = split_file_for_preview(saved_path, chunk_size, chunk_overlap, strategy)
                        raw = read_raw_for_preview(saved_path)
                        if not chunks:
                            if saved_path.exists():
                                saved_path.unlink()
                            st.warning("文件无文本内容或无法切分，请检查文件。")
                        else:
                            # 新预览覆盖旧待入库状态（旧临时文件清理）
                            if pending:
                                _clear_pending()
                            st.session_state.kb_pending = {
                                "name": uploaded_file.name,
                                "hash": current_hash,
                                "path": str(saved_path),
                                "chunks": chunks,
                                "chunk_size": chunk_size,
                                "chunk_overlap": chunk_overlap,
                                "strategy": strategy,
                                "raw": raw,
                            }
                            st.rerun()
                    except Exception as e:
                        logger.error(f"切分预览失败 name={uploaded_file.name}: {e}")
                        st.error("切分失败，请检查文件是否损坏，或查看服务日志。")

            if pending and col_cancel.button("✗ 取消预览", use_container_width=True):
                _clear_pending()
                st.rerun()

            # 入库按钮：始终贴在顶部，参数变更或空间守卫失败时禁用
            if pending:
                _check_pending_params_match = (
                    pending["chunk_size"] == chunk_size
                    and pending["chunk_overlap"] == chunk_overlap
                    and pending.get("strategy", "QA对切分") == strategy
                )
                if not _check_pending_params_match:
                    st.warning("切分参数已修改，请点击上方「重新生成切分预览」后再入库。")
                if st.button(
                    "✅ 确认切分效果，向量化入库",
                    type="primary",
                    use_container_width=True,
                    key="kb_confirm_top",
                    disabled=not _check_pending_params_match or not _KB_OK,
                ):
                    _do_confirm_ingest(pending, _KB_OK)

            # 顶部已展示入库按钮；预览区只做展示，不再重复
            pending = st.session_state.get("kb_pending")
            if pending:
                chunks = pending["chunks"]
                lengths = [len(c.page_content) for c in chunks]
                params_match = (
                    pending["chunk_size"] == chunk_size
                    and pending["chunk_overlap"] == chunk_overlap
                    and pending.get("strategy", "QA对切分") == strategy
                )
                raw = pending.get("raw") or {"kind": "text", "text": "", "df": None, "sheet_names": []}

                st.success(
                    f"✅ 预览已生成（{strategy}）：共 **{len(chunks)}** 个块"
                    f"（平均 {sum(lengths)//len(lengths)} 字符，"
                    f"最短 {min(lengths)}，最长 {max(lengths)}）"
                )

                # 视图模式切换：双视图（左右对照）/ 仅切分（兼容旧习惯）
                view_mode = st.radio(
                    "预览视图",
                    ["👈 原文 + 切分 双视图", "📑 仅切分结果"],
                    horizontal=True,
                    key="kb_view_mode",
                    label_visibility="collapsed",
                )

                if view_mode == "📑 仅切分结果":
                    st.subheader("切分效果预览", divider="gray")
                    show_n = min(len(chunks), MAX_PREVIEW_CHUNKS)
                    for i, chunk in enumerate(chunks[:show_n], start=1):
                        with st.container(border=True):
                            st.markdown(f"**📄 块 {i}** · 长度 {lengths[i-1]} 字符")
                            if strategy == "表格行切分" and chunk.metadata.get("is_table_split"):
                                _render_table_row_card(i, chunk)
                            else:
                                st.text(chunk.page_content)
                    if len(chunks) > MAX_PREVIEW_CHUNKS:
                        st.caption(f"仅展示前 {MAX_PREVIEW_CHUNKS} 块，入库时会写入全部 {len(chunks)} 块。")
                else:
                    # 双视图：左侧原文，右侧切分卡片（带起止位置高亮）
                    col_raw, col_chunks = st.columns([1, 1], gap="medium")
                    with col_raw:
                        st.markdown("##### 📄 文档原文")
                        if raw.get("kind") == "table" and raw.get("df") is not None:
                            sheets = raw.get("sheet_names") or []
                            if len(sheets) > 1:
                                sheet = st.selectbox("Sheet", sheets, key="kb_raw_sheet")
                                import pandas as pd
                                df_show = pd.read_excel(str(pending["path"]), sheet_name=sheet).fillna("")
                                st.dataframe(df_show, use_container_width=True, height=520)
                                st.caption(f"{sheet}：{len(df_show)} 行 × {len(df_show.columns)} 列")
                            else:
                                st.dataframe(raw["df"], use_container_width=True, height=520)
                                st.caption(f"共 {len(raw['df'])} 行 × {len(raw['df'].columns)} 列")
                        else:
                            text = raw.get("text", "")
                            if text:
                                st.text_area(
                                    "原文内容",
                                    value=text,
                                    height=520,
                                    label_visibility="collapsed",
                                    disabled=True,
                                    key="kb_raw_text",
                                )
                            else:
                                st.info("该文件无文本内容或读取失败")

                    with col_chunks:
                        st.markdown(f"##### ✂️ 切分结果（{len(chunks)} 块）")
                        show_n = min(len(chunks), MAX_PREVIEW_CHUNKS)
                        for i, chunk in enumerate(chunks[:show_n], start=1):
                            sc = chunk.metadata.get("start_char", -1)
                            ec = chunk.metadata.get("end_char", -1)
                            ri = chunk.metadata.get("row_index", -1)
                            pos_tag = f" · 原文 [{sc}, {ec}]" if sc >= 0 else ""
                            row_tag = f" · 第 {ri + 1} 行" if ri >= 0 else ""
                            with st.container(border=True):
                                st.markdown(f"**📄 块 {i}** · 长度 {lengths[i-1]} 字符{pos_tag}{row_tag}")
                                if strategy == "表格行切分" and chunk.metadata.get("is_table_split"):
                                    _render_table_row_card(i, chunk)
                                else:
                                    st.text(chunk.page_content)
                        if len(chunks) > MAX_PREVIEW_CHUNKS:
                            st.caption(f"仅展示前 {MAX_PREVIEW_CHUNKS} 块，入库时会写入全部 {len(chunks)} 块。")
                        # 顶部已有入库按钮，这里给一个轻提示引导用户回到顶部
                        st.info("👆 切分确认无误后，请回到页面顶部点击「确认切分效果，向量化入库」")

# =========================================================
# Tab 2：已入库文档管理
# =========================================================
with tab_files:
    file_records = get_all_file_records()
    st.caption(f"当前知识库共 **{len(file_records)}** 个文件")
    if not file_records:
        st.info("知识库为空，请先在「上传与切分预览」标签页上传文档。")
    else:
        for file_hash, record in sorted(
            file_records.items(), key=lambda x: x[1]["uploaded_at"], reverse=True
        ):
            with st.expander(
                f"📄 {record['source']}（{record['chunk_count']} 块）", expanded=False
            ):
                col_meta1, col_meta2 = st.columns(2)
                col_meta1.caption(f"入库时间：{record['uploaded_at']}")
                col_meta2.caption(f"Hash：{record['file_hash'][:12]}...")

                with st.expander("🔍 查看切块详情（前 20 块）", expanded=False):
                    try:
                        chunks = get_chunks_for_file(file_hash, limit=20)
                        st.caption(
                            f"显示前 {len(chunks)} 个块（共 {record['chunk_count']} 个）"
                        )
                        for i, chunk in enumerate(chunks, start=1):
                            with st.expander(
                                f"块 {i}（长度 {len(chunk.page_content)} 字符）"
                            ):
                                st.text(chunk.page_content)
                    except Exception as e:
                        logger.error(f"读取切块失败 hash={file_hash}: {e}")
                        st.error("读取切块失败，请查看服务日志。")

                st.divider()
                # 删除二次确认
                if st.session_state.get("kb_pending_delete") == file_hash:
                    st.warning(f"⚠️ 确认删除「{record['source']}」及其全部 {record['chunk_count']} 个块？")
                    col_ok, col_cancel = st.columns(2)
                    if col_ok.button("✓ 确认删除", key=f"kb_del_ok_{file_hash}", type="primary"):
                        success = delete_file_from_vectorstore(
                            file_hash, record["stored_path"]
                        )
                        st.session_state.kb_pending_delete = None
                        if success:
                            bump_kb_version()
                            build_bm25_index.clear()
                            st.success(f"已删除：{record['source']}")
                            st.rerun()
                        else:
                            st.error("删除失败，请查看服务日志。")
                    if col_cancel.button("✗ 取消", key=f"kb_del_cancel_{file_hash}"):
                        st.session_state.kb_pending_delete = None
                        st.rerun()
                else:
                    if st.button(
                        "🗑️ 删除此文件", key=f"kb_del_{file_hash}", use_container_width=True
                    ):
                        st.session_state.kb_pending_delete = file_hash
                        st.rerun()
