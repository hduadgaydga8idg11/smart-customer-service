# -*- coding: utf-8 -*-
"""
检索模块：向量检索 / BM25 关键词检索 / RRF 混合检索 / Rerank 精排 /
文档切分 / 向量库操作（从 智能客服助手.py 拆出）

依赖注入：主程序加载 vectorstore 后调用 set_vectorstore(vs) 注入。
"""
import hashlib
import logging
import os
import re
from pathlib import Path

import jieba
import numpy as np
import streamlit as st
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter, MarkdownHeaderTextSplitter
from rank_bm25 import BM25Okapi

logger = logging.getLogger("rag_app")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DOCS_DIR = PROJECT_ROOT / "docs"
DOCS_DIR.mkdir(exist_ok=True)
MAX_FILE_SIZE_MB = 50  # 上传文件大小上限（MB），知识库页预校验与提示共用

# =========================================================
# 模块级状态：vectorstore 由主程序注入
# =========================================================
_vectorstore = None


def set_vectorstore(vs):
    """主程序加载 Chroma 后调用此函数注入 vectorstore"""
    global _vectorstore
    _vectorstore = vs


def _get_vs():
    """获取已注入的 vectorstore（未注入时报错）"""
    if _vectorstore is None:
        raise RuntimeError("vectorstore 尚未初始化，请先调用 set_vectorstore()")
    return _vectorstore


# =========================================================
# Rerank 模型配置
# =========================================================
os.environ.setdefault("HF_ENDPOINT", "https://hf-mirror.com")

RERANK_MODEL_OPTIONS = {
    "bge-reranker-base（更快，CPU 友好）": "BAAI/bge-reranker-base",
    "bge-reranker-v2-m3（更准，效果更好）": "BAAI/bge-reranker-v2-m3",
}

LOCAL_RERANK_DIRS = {
    "BAAI/bge-reranker-base": PROJECT_ROOT / "models" / "bge-reranker-base",
    "BAAI/bge-reranker-v2-m3": PROJECT_ROOT / "models" / "bge-reranker-v2-m3",
}


def resolve_rerank_model_path(model_id: str) -> str:
    """若本地模型目录存在且包含权重文件，则返回本地路径，否则返回 HF 模型 ID"""
    local_dir = LOCAL_RERANK_DIRS.get(model_id)
    if local_dir and (local_dir / "model.safetensors").exists():
        return str(local_dir)
    return model_id


# =========================================================
# 文档加载（Docling / TextLoader / PPT 降级）
# =========================================================
# Docling 复杂文档解析开关（.env：ENABLE_DOCLING，默认开启）
# 关闭后知识库仅支持 TXT/Markdown，且不会触发 docling 重依赖的加载
DOCLING_ENABLED = (os.environ.get("ENABLE_DOCLING") or "true").strip().lower() not in (
    "false", "0", "no", "off",
)
DOCLING_TYPES = [".pdf", ".docx", ".pptx", ".xlsx", ".xls", ".html", ".htm", ".csv", ".xml",
                 ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]


def _reject_legacy_ppt(file_path: Path):
    """旧版 .ppt（OLE 二进制格式）明确拒绝：解析成功率低且易产生垃圾占位文档入库。
    引导用户另存为 .pptx 后再上传。"""
    raise ValueError(
        f"不支持旧版 .ppt 格式：{file_path.name}。请用 PowerPoint/WPS 另存为 .pptx 后重新上传。"
    )


def load_with_docling(file_path: Path, max_pages: int = 200):
    """使用 Docling 高质量解析复杂文档，对图片/PDF 启用 OCR"""
    try:
        from docling.document_converter import DocumentConverter

        suffix = file_path.suffix.lower()
        if suffix == ".ppt":
            _reject_legacy_ppt(file_path)
        format_options = {}

        if suffix in [".pdf", ".png", ".jpg", ".jpeg", ".tiff", ".bmp"]:
            format_options["ocr"] = {"lang": ["chi_sim", "en"]}
            if suffix == ".pdf":
                format_options["pdf"] = {"max_num_pages": max_pages}

        converter = DocumentConverter(format_options=format_options)
        doc_result = converter.convert(str(file_path), raises_on_error=False)

        if not doc_result or not hasattr(doc_result, "document") or doc_result.document is None:
            return [Document(
                page_content="[文件转换失败，无法获取文档对象]",
                metadata={"source": file_path.name, "parse_status": "error"},
            )]

        try:
            text = doc_result.document.export_to_markdown()
        except AttributeError:
            text = ""

        if text and text.strip():
            page_count = len(doc_result.document.pages) if hasattr(doc_result.document, "pages") else 0
            return [Document(
                page_content=text,
                metadata={"source": file_path.name, "parse_status": "success", "page_count": page_count},
            )]

        return [Document(
            page_content="[文件内容为空]",
            metadata={"source": file_path.name, "parse_status": "empty"},
        )]

    except ImportError:
        raise ImportError("需要安装 docling 依赖：pip install -r requirements.txt（含 docling 及其 OCR 引擎）")
    except ValueError:
        raise  # .ppt 明确拒绝的提示原样抛出
    except Exception as e:
        raise Exception(f"Docling 处理失败：{str(e)}")


def load_file_documents(file_path: Path):
    """根据文件类型选择加载器"""
    suffix = file_path.suffix.lower()
    if suffix == ".ppt":
        _reject_legacy_ppt(file_path)
    try:
        if suffix == ".txt":
            from langchain_community.document_loaders import TextLoader
            loader = TextLoader(str(file_path), encoding="utf-8", autodetect_encoding=True)
            loaded = loader.load()
            if not loaded or not loaded[0].page_content.strip():
                return [Document(page_content="[文件内容为空]", metadata={"source": file_path.name})]
            return loaded

        elif suffix == ".md":
            with open(file_path, "r", encoding="utf-8", errors="ignore") as f:
                text = f.read()
            if not text.strip():
                return [Document(page_content="[文件内容为空]", metadata={"source": file_path.name})]
            return [Document(page_content=text, metadata={"source": file_path.name})]

        elif suffix in DOCLING_TYPES:
            if not DOCLING_ENABLED:
                raise ValueError(
                    "Docling 解析功能已停用（仅支持 TXT/Markdown）；"
                    "如需 PDF/Word/图片等格式，在 .env 设 ENABLE_DOCLING=true 后重启应用"
                )
            return load_with_docling(file_path)

        else:
            raise ValueError(f"暂不支持该文件类型：{suffix}")
    except Exception as e:
        raise Exception(f"读取 {suffix} 文件失败：{str(e)}")


def read_raw_for_preview(file_path: Path) -> dict:
    """为左侧「原文视图」读取原始内容，返回 dict：
        - kind: "table"（Excel/CSV 多行多列）/ "text"（其他格式纯文本）
        - df: pandas.DataFrame（kind=table 时）
        - text: str（kind=text 时，已是 Markdown/PDF 提取后的可读文本）
        - sheet_names: list[str]（Excel 多 sheet 时供切换）
    设计原则：
        - 与入库流程一致：Excel/CSV 用 pandas 原样读取（保留所有行列），不做切分
        - PDF/Word/PPT 等用 Docling 导出 markdown 作为「原文」展示
        - 加载失败不抛异常，降级为空文本，避免阻塞切分预览
    """
    suffix = file_path.suffix.lower()
    empty = {"kind": "text", "text": "", "df": None, "sheet_names": []}
    try:
        if suffix in [".xlsx", ".xls"]:
            try:
                import pandas as pd
                xls = pd.ExcelFile(str(file_path))
                sheets = xls.sheet_names or ["Sheet1"]
                first = sheets[0]
                df = xls.parse(first).fillna("")
                return {"kind": "table", "df": df, "sheet_names": sheets, "text": ""}
            except Exception as e:
                logger.warning(f"读取 Excel 原文失败，降级为文本 source={file_path.name}: {e}")
                return empty
        elif suffix == ".csv":
            try:
                import pandas as pd
                df = pd.read_csv(str(file_path), encoding="utf-8", encoding_errors="ignore").fillna("")
                return {"kind": "table", "df": df, "sheet_names": [], "text": ""}
            except Exception as e:
                logger.warning(f"读取 CSV 原文失败，降级为文本 source={file_path.name}: {e}")
                return empty
        else:
            docs = load_file_documents(file_path)
            text = "\n\n".join(d.page_content for d in docs) if docs else ""
            return {"kind": "text", "text": text, "df": None, "sheet_names": []}
    except Exception as e:
        logger.warning(f"读取原文失败 source={file_path.name}: {e}")
        return empty


# =========================================================
# 文件工具函数
# =========================================================
def calculate_file_hash(file_bytes: bytes) -> str:
    """计算文件内容的 SHA256 哈希值"""
    return hashlib.sha256(file_bytes).hexdigest()


def safe_filename(original_name: str) -> str:
    """净化文件名，防止路径穿越（同时处理 / 和 \\ 分隔符，兼容 Windows 路径）"""
    return Path(str(original_name).replace("\\", "/")).name


def save_uploaded_file(uploaded_file, file_hash: str) -> Path:
    """保存上传文件到 docs 目录，文件名添加哈希前缀"""
    safe_name = safe_filename(uploaded_file.name)
    save_path = DOCS_DIR / f"{file_hash[:12]}_{safe_name}"
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())
    return save_path


# =========================================================
# 向量库操作
# =========================================================
def get_all_file_records(vs=None):
    """从 Chroma 中读取全部文件记录（按 file_hash 聚合）"""
    if vs is None:
        vs = _get_vs()
    data = vs.get(include=["metadatas"])
    records = {}
    for metadata in data.get("metadatas", []):
        if not metadata:
            continue
        fh = metadata.get("file_hash")
        if not fh:
            continue
        if fh not in records:
            records[fh] = {
                "source": metadata.get("source", "未知文件"),
                "file_hash": fh,
                "uploaded_at": metadata.get("uploaded_at", "未知时间"),
                "stored_path": metadata.get("stored_path", ""),
                "chunk_count": 0,
            }
        records[fh]["chunk_count"] += 1
    return records


def is_file_already_in_db(file_hash: str, vs=None) -> bool:
    """检查文件哈希是否已存在"""
    records = get_all_file_records(vs)
    return file_hash in records


def get_chunks_for_file(file_hash: str, limit: int = None, vs=None):
    """获取某个文件的所有切块"""
    if vs is None:
        vs = _get_vs()
    results = vs.get(where={"file_hash": file_hash}, include=["documents", "metadatas"])
    docs = []
    for content, metadata in zip(results["documents"], results["metadatas"]):
        docs.append(Document(page_content=content, metadata=metadata))
    docs.sort(key=lambda x: x.metadata.get("chunk_id", 0))
    if limit:
        docs = docs[:limit]
    return docs


def delete_file_from_vectorstore(file_hash: str, stored_path: str, vs=None) -> bool:
    """删除指定文件的所有向量块及本地文件（仅允许删除 docs 目录内的文件）"""
    if vs is None:
        vs = _get_vs()
    try:
        vs.delete(where={"file_hash": file_hash})
        if stored_path:
            path = Path(stored_path).resolve()
            if path.is_relative_to(DOCS_DIR.resolve()):
                if path.exists():
                    path.unlink()
                    logger.info(f"已删除本地文件: {path.name}")
            else:
                logger.warning(f"拒绝删除 docs 目录外的文件: {path}")
        return True
    except Exception as e:
        logger.error(f"删除向量数据失败 hash={file_hash}: {e}")
        try:
            st.error("删除失败，请查看服务日志定位原因。")
        except Exception:
            pass
        return False


def add_prepared_chunks_to_vectorstore(
    chunks: list,
    file_path: Path,
    original_file_name: str,
    file_hash: str,
    vs=None,
) -> tuple:
    """将预览确认后的切块直接写入向量库（切块已在预览阶段生成，保证所见即所入）。"""
    if vs is None:
        vs = _get_vs()
    if is_file_already_in_db(file_hash, vs):
        if file_path.exists():
            file_path.unlink()
        return False, "该文件内容已经在知识库中，不能重复入库。"
    from datetime import datetime
    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata = _sanitize_chroma_metadata({
            **chunk.metadata,
            "source": original_file_name,
            "file_hash": file_hash,
            "chunk_id": idx,
            "uploaded_at": uploaded_at,
            "stored_path": str(file_path),
        })
        ids.append(f"{file_hash}_{idx}")
    try:
        vs.add_documents(documents=chunks, ids=ids)
    except Exception as e:
        if file_path.exists():
            file_path.unlink()
        logger.error(f"向量库写入失败 name={original_file_name}: {e}")
        return False, "向量库写入失败，请检查模型服务是否正常后重试。"
    logger.info(f"文件入库成功 name={original_file_name} chunks={len(chunks)}（预览确认流程）")
    return True, f"入库成功，共写入 {len(chunks)} 个文档块。"


# =========================================================
# 知识库来源管理
# =========================================================
def get_kb_sources() -> list[str]:
    """知识库中所有来源文件名（排序后）"""
    records = get_all_file_records()
    return sorted(r["source"] for r in records.values())


def get_kb_source_counts() -> dict[str, int]:
    """来源文件名 → 块数"""
    records = get_all_file_records()
    return {r["source"]: r["chunk_count"] for r in records.values()}


def get_active_sources() -> set | None:
    """当前勾选参与检索的来源集合：
    - 全选或未初始化 → None（不过滤，全库检索）
    - 部分勾选 → 对应集合
    - 全部取消 → 空集合（检索直接返回空）
    """
    all_sources = set(get_kb_sources())
    selected = st.session_state.get("kb_selected")
    if selected is None:
        return None
    selected = {s for s in selected if s in all_sources}
    if selected == all_sources:
        return None
    return selected


# =========================================================
# 切分策略
# =========================================================
SPLIT_STRATEGIES = ["QA对切分", "标题切分", "语义切分", "表格行切分"]


def _split_recursive(documents, chunk_size, chunk_overlap):
    """递归字符切分：按段落→句子→标点优先级递归切分（默认策略）"""
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    return splitter.split_documents(documents)


def _split_markdown(documents, max_chunk_size, source_name):
    """Markdown 标题切分：按 #/##/### 标题分节，超长节二次切分"""
    md_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=[("#", "h1"), ("##", "h2"), ("###", "h3")],
        strip_headers=False,
    )
    chunks = []
    for doc in documents:
        text = doc.page_content
        try:
            md_chunks = md_splitter.split_text(text)
        except Exception as e:
            logger.warning(f"Markdown 切分失败，降级为递归字符切分 source={source_name}: {e}")
            return _split_recursive(documents, max_chunk_size, 0)
        cursor = 0
        for chunk in md_chunks:
            header_path = _strip_md_header(" > ".join(v for v in chunk.metadata.values() if v))
            chunk_body = chunk.page_content
            # 在原文中定位该块的起止位置（用于左右对照视图高亮）
            start_char = text.find(chunk_body, cursor) if chunk_body else -1
            if start_char < 0:
                start_char = cursor
            end_char = start_char + len(chunk_body)
            cursor = end_char
            if header_path and not chunk_body.startswith(header_path):
                text_out = f"{header_path}\n{chunk_body}"
                full_start = start_char
            else:
                text_out = chunk_body
                full_start = start_char
            if len(text_out) > max_chunk_size:
                sub_splitter = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                )
                sub_docs = sub_splitter.split_text(text_out)
                offset = full_start
                for sd in sub_docs:
                    pos = text_out.find(sd)
                    sd_start = (offset + pos) if pos >= 0 else -1
                    sd_end = sd_start + len(sd) if sd_start >= 0 else -1
                    chunks.append(Document(
                        page_content=sd,
                        metadata={
                            **chunk.metadata,
                            "header_path": header_path,
                            "source": source_name,
                            "start_char": sd_start,
                            "end_char": sd_end,
                        },
                    ))
                    if pos >= 0:
                        offset = offset + pos + len(sd)
            else:
                chunks.append(Document(
                    page_content=text_out,
                    metadata={
                        **chunk.metadata,
                        "header_path": header_path,
                        "source": source_name,
                        "start_char": full_start,
                        "end_char": full_start + len(text_out),
                    },
                ))
    return chunks if chunks else _split_recursive(documents, max_chunk_size, 0)


_TOKENIZER = None


def _get_tokenizer():
    """获取 bge-m3 tokenizer（缓存，仅加载一次）"""
    global _TOKENIZER
    if _TOKENIZER is not None:
        return _TOKENIZER
    try:
        from transformers import AutoTokenizer
        _TOKENIZER = AutoTokenizer.from_pretrained("BAAI/bge-m3")
    except Exception as e:
        logger.warning(f"加载 tokenizer 失败，降级为字符计数: {e}")
        _TOKENIZER = None
    return _TOKENIZER


# QA 对切分：匹配【问题】标记（兼容【】内不含换行）
_QA_PATTERN = re.compile(r"【[^】\n]+】")


# 块级标签：转换时在子节点前后补换行，保证段落/列表结构清晰
_BLOCK_TAGS = {"p", "div", "section", "article", "header", "footer",
               "ul", "ol", "li", "h1", "h2", "h3", "h4", "h5", "h6",
               "pre", "blockquote", "tr", "table"}


_MD_HEADER_PATTERN = re.compile(r"^#{1,6}\s*")


def _strip_md_header(s: str) -> str:
    """去掉行首 Markdown # 符号（如 '# 一级标题' → '一级标题'），保留标题文字。

    切分后块文本里拼接章节标题作为上下文前缀时使用，避免 # 号污染向量检索。
    """
    if not s:
        return ""
    # 去掉每行行首的 # 号（多行 header_path 也安全）
    return "\n".join(_MD_HEADER_PATTERN.sub("", line).strip() for line in s.split("\n") if line.strip()).strip()


_STRIP_HEADER_LINE = re.compile(r"^#{1,6}\s+.*\n?", flags=re.MULTILINE)


def _strip_md_header_lines(text: str) -> str:
    """剥除文本里所有「整行的 Markdown 标题行」（如 '## 二、发货与物流'）。

    用于 _split_qa 切分边界带进来的下一个章节标题，避免 # 号残留污染向量。
    不会影响正文里的 # 符号（因为正文里的 # 不会独占整行）。
    """
    if not text or "#" not in text:
        return text
    cleaned = _STRIP_HEADER_LINE.sub("", text)
    # 合并因剥除产生的连续空行（最多保留一个空行）
    cleaned = re.sub(r"\n{3,}", "\n\n", cleaned)
    return cleaned.strip()


def _html_to_markdown(html: str) -> str:
    """把 HTML 片段转为 markdown（保留语义，去掉标签噪音）。

    设计目标：客服场景下知识库入库清洗。
        - 入库前去掉 `<p>`/`<a>`/`<span>` 等标签字面量，避免向量被标签污染；
        - 同时把链接/加粗/列表/换行等格式语义保留为 markdown，让下游
          `st.markdown()` 渲染时可点击、可换行、可加粗。
        - 不引入额外依赖（用项目已有的 beautifulsoup4）。

    支持的标签：
        <a href="URL">text</a>     → [text](URL)    （链接可点击跳转）
        <strong>/<b>text</strong>  → **text**        （加粗）
        <em>/<i>text</em>          → *text*          （斜体）
        <code>text</code>          → `text`          （行内代码）
        <pre>text</pre>            → ```text```      （代码块）
        <h1>~ <h6>text</h?>        → # ~ ###### text （标题）
        <ul><li>x</li></ul>        → - x             （无序列表）
        <ol><li>x</li></ol>        → 1. x            （有序列表）
        <br>                       → 换行
        <p>/<div> 等块级           → 子节点 + 换行分隔
        其他标签                    → 去掉标签、仅保留文本
    """
    if not html or "<" not in html:
        return html
    try:
        from bs4 import BeautifulSoup, NavigableString
    except ImportError:
        # 兜底：纯正则去标签（不推荐，但保证可运行）
        return re.sub(r"<[^>]+>", "", html)

    soup = BeautifulSoup(html, "html.parser")

    def _render_inline(node) -> str:
        """处理内联节点（无换行）"""
        if isinstance(node, NavigableString):
            return str(node)
        if not hasattr(node, "name") or node.name is None:
            return ""
        name = (node.name or "").lower()
        if name == "br":
            return "\n"
        if name == "a":
            href = node.get("href", "") or ""
            text = "".join(_render_inline(c) for c in node.children).strip()
            if not href or not text:
                return text
            return f"[{text}]({href})"
        if name in ("strong", "b"):
            inner = "".join(_render_inline(c) for c in node.children).strip()
            return f"**{inner}**" if inner else ""
        if name in ("em", "i"):
            inner = "".join(_render_inline(c) for c in node.children).strip()
            return f"*{inner}*" if inner else ""
        if name == "code":
            inner = "".join(_render_inline(c) for c in node.children)
            return f"`{inner}`" if inner else ""
        if name in ("p", "div", "span", "section", "article"):
            return "".join(_render_inline(c) for c in node.children)
        # 其他标签（含未知）：去掉标签，保留子节点文本
        return "".join(_render_inline(c) for c in node.children)

    def _render_block(node) -> str:
        """处理块级节点（必要时补换行）"""
        if isinstance(node, NavigableString):
            text = str(node)
            return text
        if not hasattr(node, "name") or node.name is None:
            return ""
        name = (node.name or "").lower()
        if name == "br":
            return "\n"
        if name in ("h1", "h2", "h3", "h4", "h5", "h6"):
            level = int(name[1])
            text = "".join(_render_inline(c) for c in node.children).strip()
            return f"\n\n{'#' * level} {text}\n\n" if text else ""
        if name == "p":
            inner = "".join(_render_inline(c) for c in node.children).strip()
            return f"\n\n{inner}\n\n" if inner else "\n"
        if name == "ul":
            items = []
            for li in node.find_all("li", recursive=False):
                inner = "".join(_render_inline(c) for c in li.children).strip()
                if inner:
                    items.append(f"- {inner}")
            return ("\n\n" + "\n".join(items) + "\n\n") if items else ""
        if name == "ol":
            items = []
            for idx, li in enumerate(node.find_all("li", recursive=False), start=1):
                inner = "".join(_render_inline(c) for c in li.children).strip()
                if inner:
                    items.append(f"{idx}. {inner}")
            return ("\n\n" + "\n".join(items) + "\n\n") if items else ""
        if name == "pre":
            inner_text = node.get_text()
            return f"\n\n```\n{inner_text}\n```\n\n" if inner_text.strip() else ""
        if name == "blockquote":
            inner = _render_children(node).strip()
            if not inner:
                return ""
            quoted = "\n".join(f"> {line}" for line in inner.split("\n"))
            return f"\n\n{quoted}\n\n"
        if name == "li":
            # 顶层 li（不在 ul/ol 里）：当内联处理
            return _render_inline(node)
        # 其他块级（含 div/section）：递归处理子节点
        return _render_children(node)

    def _render_children(node) -> str:
        out = []
        for c in node.children:
            if isinstance(c, NavigableString):
                txt = str(c)
                if txt.strip():
                    out.append(txt)
            elif c.name in _BLOCK_TAGS:
                out.append(_render_block(c))
            else:
                out.append(_render_inline(c))
        return "".join(out)

    md = _render_children(soup)
    # 规范化：把连续 3 个及以上换行压成 2 个，修剪首尾空白
    md = re.sub(r"\n{3,}", "\n\n", md)
    return md.strip()


def _sanitize_chroma_metadata(meta: dict) -> dict:
    """入库前清洗 metadata，剔除 ChromaDB 不接受的值。

    ChromaDB 限制：
        - 顶层值只允许 str/int/float/bool/非空 list；
        - 空 list / None / 空字符串会让 upsert 抛错（Expected metadata list value
          ... to be non-empty in upsert）。
    本函数把空 list/None/空字符串直接丢弃，保留其他字段。
    """
    out = {}
    for k, v in meta.items():
        if v is None:
            continue
        if isinstance(v, str):
            if v == "":
                continue
            out[k] = v
        elif isinstance(v, list):
            if not v:
                continue
            out[k] = v
        else:
            out[k] = v
    return out


def _row_text_from_table(df, row_idx: int) -> dict:
    """把一行所有列拆为结构化字段（同时返回拼接好的 page_content）。

    智能识别三列常见结构：知识标题 / 相似问法 / 答案内容。
    其他列统一并入「其他信息」。这样既能用结构化字段做 UI 横排展示，
    也能把拼接好的文本送去做向量化检索。
    返回：{"text": str, "title": str, "questions": str, "answer": str, "others": list[str]}
    """
    import pandas as pd
    row = df.iloc[row_idx]
    cells = [(str(c).strip(), row[c]) for c in df.columns]
    cells = [(label, val) for label, val in cells if pd.notna(val) and str(val).strip()]

    def _norm(label: str) -> str:
        return label.replace(" ", "").replace("\n", "").lower()

    title = ""
    questions = ""
    answer = ""
    others = []
    for label, val in cells:
        nl = _norm(label)
        v = str(val).strip()
        if not v:
            continue
        if any(k in nl for k in ["标题", "分类", "主题", "知识", "title", "category"]):
            title = v
        elif any(k in nl for k in ["相似问法", "问法", "相似问", "类似问", "相似问题", "问题", "question", "qa"]):
            questions = v
        elif any(k in nl for k in ["答案", "回答", "回复", "答复", "内容", "详情", "answer", "response"]):
            # 答案列清洗 HTML 标签 → 保留 markdown 语义（链接/加粗/换行/列表）
            # 若不含标签，原样返回（保留多行文本、保留全角空格等格式）
            answer = _html_to_markdown(v) if "<" in v else v
        else:
            others.append(f"{label}：{v}")

    parts = []
    if title:
        parts.append(f"知识标题：{title}")
    if questions:
        parts.append(f"相似问法：{questions}")
    if answer:
        parts.append(f"答案内容：{answer}")
    parts.extend(others)
    text = "\n".join(parts)
    return {"text": text, "title": title, "questions": questions, "answer": answer, "others": others}


def _split_table_row(file_path: Path, max_chunk_size: int, source_name: str) -> list:
    """表格行切分：xlsx/xls/csv 按行切分，每行 = 一个知识块。

    适用于「知识标题 / 相似问法 / 答案内容」三列结构的标准 FAQ 表格。
    行为约定：
        ① 用 pandas 直接读取，跳过 Docling（避免表格被转成 markdown 破坏结构）；
        ② 第一行作为表头，每一行（除空行）合并为一个块；
        ③ 块文本按「知识标题 / 相似问法 / 答案内容」结构化拼接，便于检索匹配；
        ④ 单元格有换行的"相似问法"按换行展开为多个并列问法，提升问法召回；
        ⑤ 超长行按 max_chunk_size 二次切分（兜底）。
    """
    import pandas as pd
    suffix = file_path.suffix.lower()
    try:
        if suffix == ".csv":
            df = pd.read_csv(str(file_path), encoding="utf-8", encoding_errors="ignore").fillna("")
        elif suffix in [".xlsx", ".xls"]:
            xls = pd.ExcelFile(str(file_path))
            df = xls.parse(xls.sheet_names[0]).fillna("")
        else:
            raise ValueError(f"表格行切分仅支持 xlsx/xls/csv，当前文件类型：{suffix}")
    except Exception as e:
        logger.warning(f"读取表格失败，降级为递归字符切分 source={source_name}: {e}")
        docs = load_file_documents(file_path)
        return _split_recursive(docs, max_chunk_size, 0)

    chunks = []
    n = len(df)
    for i in range(n):
        parsed = _row_text_from_table(df, i)
        text = parsed["text"]
        if not text:
            continue
        meta_base = _sanitize_chroma_metadata({
            "source": source_name,
            "row_index": i,
            "row_total": n,
            "header": list(df.columns),
            "title": parsed["title"],
            "questions": parsed["questions"],
            "answer": parsed["answer"],
            "others": parsed["others"],
            "is_table_split": True,
        })
        if len(text) <= max_chunk_size:
            chunks.append(Document(page_content=text, metadata=meta_base))
        else:
            sub = RecursiveCharacterTextSplitter(
                chunk_size=max_chunk_size,
                chunk_overlap=0,
                separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
            ).split_text(text)
            for sd in sub:
                chunks.append(Document(page_content=sd, metadata={**meta_base}))
    if not chunks:
        docs = load_file_documents(file_path)
        return _split_recursive(docs, max_chunk_size, 0)
    return chunks


def _split_qa(documents, max_chunk_size, source_name):
    """QA 对切分：按【问题】标记切分，一个「问题+答案」= 一个块。

    客服 FAQ 的典型结构是「## 章节标题」下挂多个「【问题】答案」。
    本实现：① 每个【问题】到下一个【问题】（或文末）切为一块；② 把最近的
    ## 标题作为上下文前缀拼入该块（保证块自包含、不丢章节语境）；③ 超长
    问答对按 max_chunk_size 二次切分；④ 整篇识别不到【】结构时降级为标题切分。
    """
    chunks = []
    for doc in documents:
        text = doc.page_content
        matches = list(_QA_PATTERN.finditer(text))
        if not matches:
            continue  # 该文档无 QA 结构，交给下方统一降级
        # 预扫所有 Markdown 标题位置，用于给 QA 对补章节上下文
        header_positions = [
            (hm.start(), hm.group(1).strip())
            for hm in re.finditer(r"(?m)^(#{1,3}\s+.+)$", text)
        ]
        for i, m in enumerate(matches):
            start = m.start()
            end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
            qa_text = text[start:end].strip()
            if not qa_text:
                continue
            # 找 start 之前最近的标题作为上下文前缀（去掉 # 号，仅保留标题文字）
            ctx = ""
            for pos, h in header_positions:
                if pos < start:
                    ctx = _strip_md_header(h)
                else:
                    break
            ctx_offset = 0
            if ctx and not qa_text.startswith(ctx):
                qa_text = f"{ctx}\n{qa_text}"
                ctx_offset = len(ctx) + 1  # 标题 + 换行
            qa_start = start - ctx_offset if ctx_offset else start
            qa_end = qa_start + len(qa_text)
            # 剥除块内残留的整行标题（如下一个章节标题被切分边界带进来）
            qa_text = _strip_md_header_lines(qa_text)
            if not qa_text:
                continue
            if len(qa_text) > max_chunk_size:
                sub = RecursiveCharacterTextSplitter(
                    chunk_size=max_chunk_size,
                    chunk_overlap=0,
                    separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
                ).split_text(qa_text)
                offset = qa_start
                for sd in sub:
                    pos = qa_text.find(sd)
                    sd_start = (offset + pos) if pos >= 0 else -1
                    sd_end = sd_start + len(sd) if sd_start >= 0 else -1
                    chunks.append(Document(
                        page_content=sd,
                        metadata={
                            **doc.metadata,
                            "source": source_name,
                            "start_char": sd_start,
                            "end_char": sd_end,
                        },
                    ))
                    if pos >= 0:
                        offset = offset + pos + len(sd)
            else:
                chunks.append(Document(
                    page_content=qa_text,
                    metadata={
                        **doc.metadata,
                        "source": source_name,
                        "start_char": qa_start,
                        "end_char": qa_end,
                    },
                ))
    if not chunks:
        return _split_markdown(documents, max_chunk_size, source_name)
    return chunks


def _get_embeddings_for_split():
    """获取嵌入模型（语义切分用），从注入的 vectorstore 取；不可用返回 None。"""
    try:
        vs = _get_vs()
        return getattr(vs, "_embedding_function", None) or getattr(vs, "embedding_function", None)
    except Exception:
        return None


def _split_sentences(text: str) -> list:
    """按句子边界拆开（中文/英文句末标点 + 换行）"""
    parts = re.split(r"(?<=[。！？!?；;])\s*|\n+", text)
    return [p.strip() for p in parts if p.strip()]


def _cosine(a, b) -> float:
    """余弦相似度"""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    na, nb = np.linalg.norm(a), np.linalg.norm(b)
    if na == 0 or nb == 0:
        return 0.0
    return float(np.dot(a, b) / (na * nb))


def _split_semantic(documents, chunk_size, source_name, embeddings=None):
    """语义切分：按句子相似度在语义转折处切分，尽量凑近 chunk_size。

    流程：按句子拆 → 批量嵌入 → 算相邻句子余弦相似度 → 相似度低于阈值
    （0.5，即话题转折）或累计超过 chunk_size 时切分。嵌入不可用时降级为递归切分。
    """
    embeddings = embeddings or _get_embeddings_for_split()
    if embeddings is None:
        return _split_recursive(documents, chunk_size, 0)
    chunks = []
    threshold = 0.5  # 相邻句子相似度低于此值视为语义转折
    for doc in documents:
        text = doc.page_content
        sentences = _split_sentences(text)
        if len(sentences) <= 1:
            chunks.append(Document(page_content=text, metadata={**doc.metadata, "source": source_name}))
            continue
        try:
            vecs = embeddings.embed_documents(sentences)
        except Exception as e:
            logger.warning(f"语义切分嵌入失败，降级为递归切分 source={source_name}: {e}")
            return _split_recursive(documents, chunk_size, 0)
        sims = [_cosine(vecs[i], vecs[i + 1]) for i in range(len(vecs) - 1)]
        # 贪心聚合：相似度够且未超 chunk_size 则合并，否则切块
        groups = []
        cur = sentences[0]
        for i in range(1, len(sentences)):
            if len(cur) + len(sentences[i]) <= chunk_size and sims[i - 1] >= threshold:
                cur += sentences[i]
            else:
                groups.append(cur)
                cur = sentences[i]
        groups.append(cur)
        for g in groups:
            if g.strip():
                chunks.append(Document(page_content=g, metadata={**doc.metadata, "source": source_name}))
    return chunks if chunks else _split_recursive(documents, chunk_size, 0)


def _finalize_chunks(chunks: list) -> list:
    """切分出口处的统一清洗：所有切分策略都跑一遍，去掉残留的整行 Markdown 标题。

    设计动机：
        语义切分 / 标题切分 / 递归切分 都会让 `# ## ###` 这种标题行以「整行」
        形式残留在 page_content 里。这些符号对向量检索是噪音、对客户展示也不
        必要。统一在出口处剥除，比每个切分函数内部重复清洗更稳。
    表格行切分不需要走这里（answer 列已经过 HTML→markdown 清洗，且无标题行）。
    """
    for c in chunks:
        c.page_content = _strip_md_header_lines(c.page_content)
    return chunks


def split_file_for_preview(
    file_path: Path,
    chunk_size: int,
    chunk_overlap: int,
    strategy: str = "QA对切分",
    embeddings=None,
) -> list:
    """加载并切分文档，返回 Document 切块列表（供预览与确认入库共用）。
    strategy 决定切分方式；chunk_size 含义随策略变化（QA对/标题=超长块二次切分阈值，
    语义=目标块大小）。"""
    source_name = file_path.name
    if strategy == "表格行切分":
        return _split_table_row(file_path, chunk_size, source_name)
    documents = load_file_documents(file_path)
    if not documents:
        return []
    if strategy == "标题切分":
        chunks = _split_markdown(documents, chunk_size, source_name)
    elif strategy == "语义切分":
        chunks = _split_semantic(documents, chunk_size, source_name, embeddings=embeddings)
    else:  # QA对切分（默认）
        chunks = _split_qa(documents, chunk_size, source_name)
    return _finalize_chunks(chunks)


# =========================================================
# BM25 索引
# =========================================================
_BM25_STOPWORDS = frozenset({
    "的", "了", "是", "在", "我", "你", "他", "她", "它", "们", "这", "那", "都",
    "就", "也", "还", "又", "不", "没", "有", "和", "与", "或", "把", "被", "让",
    "给", "到", "对", "向", "从", "为", "以", "于", "及", "等", "之", "其", "此",
    "个", "要", "会", "能", "可", "可以", "什么", "怎么", "怎样", "如何",
    "吗", "呢", "吧", "啊", "哦", "嗯", "哈", "啦", "呀", "嘛", "喂",
    "一", "二", "三", "上", "下", "里", "中", "时", "地", "得",
    "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
    "in", "on", "at", "to", "for", "of", "and", "or", "not", "no", "yes",
    "i", "you", "he", "she", "it", "we", "they", "this", "that", "do", "does",
})


def tokenize_for_bm25(text: str) -> list:
    """jieba 分词（兼容中英文）+ 停用词过滤 + 单字过滤，用于 BM25 索引与查询"""
    tokens = jieba.lcut(text.lower())
    return [t.strip() for t in tokens if t.strip() and t not in _BM25_STOPWORDS and len(t.strip()) > 1]


_KB_VERSION = 0
_KB_VERSION_FILE = PROJECT_ROOT / ".kb_version"  # 版本号落盘：跨进程（Streamlit/FastAPI）可见


def _read_kb_version_file() -> int:
    try:
        return int(_KB_VERSION_FILE.read_text(encoding="utf-8").strip() or 0)
    except (OSError, ValueError):
        return 0


def bump_kb_version():
    """入库/删除/重建后递增知识库版本号，触发 BM25 索引重建。
    版本号同步写盘：api.py 独立进程读取后缓存 key 立即变化，
    避免重建知识库后其他进程的 BM25 索引滞后（原先最长 5 分钟 TTL）。"""
    global _KB_VERSION
    _KB_VERSION = max(_KB_VERSION, _read_kb_version_file()) + 1
    try:
        _KB_VERSION_FILE.write_text(str(_KB_VERSION), encoding="utf-8")
    except OSError as e:
        logger.warning(f"KB 版本号写盘失败（仅本进程内生效）: {e}")
    try:
        st.session_state.kb_version = _KB_VERSION
    except Exception:
        pass


def get_kb_version() -> int:
    """读取当前知识库版本号：磁盘文件为跨进程真源，读盘失败回退进程内值。"""
    disk_v = _read_kb_version_file()
    try:
        return max(int(st.session_state.kb_version), disk_v)
    except Exception:
        return max(_KB_VERSION, disk_v)


def _build_bm25_index_uncached(vs=None):
    """从 Chroma 读取全部文档块，构建 BM25 索引（实际构建逻辑，不走缓存）。"""
    vs = vs or _get_vs()
    data = vs.get(include=["documents", "metadatas"])
    docs = []
    for content, metadata in zip(data.get("documents", []), data.get("metadatas", [])):
        if content and content.strip():
            docs.append(Document(page_content=content, metadata=metadata or {}))
    if not docs:
        return [], None
    tokenized_corpus = [tokenize_for_bm25(d.page_content) for d in docs]
    bm25 = BM25Okapi(tokenized_corpus)
    return docs, bm25


@st.cache_data(ttl=300)
def build_bm25_index(kb_version: int):
    """从 Chroma 读取全部文档块，构建 BM25 索引（脚本上下文走缓存）。"""
    return _build_bm25_index_uncached()


def _get_bm25_index():
    """获取 BM25 索引：优先走 Streamlit 缓存；图节点工作线程无脚本上下文时降级为直接构建。"""
    try:
        return build_bm25_index(get_kb_version())
    except Exception as e:
        logger.warning(f"BM25 缓存不可用（可能处于工作线程），降级为直接构建: {e}")
        return _build_bm25_index_uncached()


def build_bm25_index_for_vectorstore(vs):
    """为指定向量库构建独立 BM25 索引，供评测专用知识库使用，不污染线上缓存。"""
    return _build_bm25_index_uncached(vs)


# =========================================================
# 三路检索
# =========================================================
def keyword_search(query: str, top_k: int, sources: set | None = None, vs=None, bm25_data=None) -> list:
    """关键词检索：BM25 算法，基于词项匹配打分。
    sources 非空时只保留指定来源文件的文档块（知识库关联过滤）。"""
    docs, bm25 = bm25_data if bm25_data is not None else (
        build_bm25_index_for_vectorstore(vs) if vs is not None else _get_bm25_index()
    )
    if bm25 is None:
        return []
    query_tokens = tokenize_for_bm25(query)
    if not query_tokens:
        return []
    scores = bm25.get_scores(query_tokens)
    candidates = [
        (i, scores[i])
        for i in range(len(scores))
        if scores[i] > 0 and (sources is None or docs[i].metadata.get("source") in sources)
    ]
    candidates.sort(key=lambda x: x[1], reverse=True)
    return [docs[i] for i, _ in candidates[:top_k]]


def vector_search(
    query: str,
    top_k: int,
    similarity_threshold: float = 0.0,
    sources: set | None = None,
    vs=None,
) -> list:
    """向量检索：基于语义相似度（bge-m3），支持相似度阈值过滤与来源文件过滤。
    similarity_threshold 为余弦相似度（0~1，越大越相似），0 表示不过滤。"""
    vs = vs or _get_vs()
    where_filter = {"source": {"$in": sorted(sources)}} if sources else None
    try:
        docs_with_scores = vs.similarity_search_with_score(
            query, k=top_k, filter=where_filter
        )
    except Exception as e:
        logger.error(f"向量检索失败: {e}")
        try:
            st.error("向量检索失败，请检查 Ollama 服务是否正常运行。")
        except Exception:
            pass
        return []
    if similarity_threshold > 0.0:
        # Chroma 返回 L2 距离（越小越相似），转换为余弦相似度近似值过滤
        # bge-m3 归一化向量空间中 cosine_sim ≈ 1 - dist/2
        return [doc for doc, score in docs_with_scores if (1.0 - score / 2.0) >= similarity_threshold]
    return [doc for doc, _ in docs_with_scores]


def dense_top_confidence(query: str, sources: set | None = None, vs=None) -> float:
    """向量路 top1 余弦近似置信度（1 - L2距离/2），无命中或异常返回 0.0。
    供置信度兜底做双信号判定：Rerank 低分但向量强命中时不予兜底。"""
    vs = vs or _get_vs()
    where_filter = {"source": {"$in": sorted(sources)}} if sources else None
    try:
        hits = vs.similarity_search_with_score(query, k=1, filter=where_filter)
        if not hits:
            return 0.0
        return max(0.0, min(1.0, 1.0 - float(hits[0][1]) / 2.0))
    except Exception as e:
        logger.warning(f"向量置信度探测失败，按 0 处理: {e}")
        return 0.0


def _doc_key(doc) -> str:
    """文档块唯一标识（用于混合检索去重）"""
    fh = doc.metadata.get("file_hash")
    cid = doc.metadata.get("chunk_id")
    if fh and cid:
        return f"{fh}_{cid}"
    return hashlib.md5(doc.page_content.encode("utf-8")).hexdigest()


def hybrid_search(
    query: str,
    top_k: int,
    similarity_threshold: float = 0.0,
    sources: set | None = None,
    vs=None,
    bm25_data=None,
) -> list:
    """混合检索：向量检索 + BM25 关键词检索，RRF（倒数排名融合）合并排序。
    sources 非空时两路检索均只在指定来源文件范围内召回（知识库关联过滤）。"""
    vec_docs = vector_search(query, top_k * 2, similarity_threshold, sources, vs=vs)
    kw_docs = keyword_search(query, top_k * 2, sources, vs=vs, bm25_data=bm25_data)
    if not vec_docs and not kw_docs:
        return []

    K = 60  # RRF 常数
    rrf_scores = {}
    doc_map = {}
    for rank, doc in enumerate(vec_docs, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (K + rank)
    for rank, doc in enumerate(kw_docs, start=1):
        key = _doc_key(doc)
        doc_map[key] = doc
        rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (K + rank)

    ranked = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
    return [doc_map[key] for key, _ in ranked[:top_k]]


def do_retrieve(
    query: str,
    top_k: int,
    similarity_threshold: float,
    retrieval_mode: str,
    sources: set | None = None,
    vs=None,
    bm25_data=None,
) -> list:
    """根据检索模式分发到对应检索器。
    sources 为非空集合时只在指定来源文件范围内召回；空集合表示无关联文档，直接返回空。"""
    if sources is not None and len(sources) == 0:
        return []
    if retrieval_mode in ("关键词检索", "关键词"):
        return keyword_search(query, top_k, sources, vs=vs, bm25_data=bm25_data)
    elif retrieval_mode == "混合检索":
        return hybrid_search(query, top_k, similarity_threshold, sources, vs=vs, bm25_data=bm25_data)
    else:
        return vector_search(query, top_k, similarity_threshold, sources, vs=vs)


# =========================================================
# Rerank 精排
# =========================================================
@st.cache_resource
def load_reranker(model_id: str):
    """加载 Rerank 交叉编码器模型（缓存，仅加载一次；优先使用本地模型目录）"""
    from sentence_transformers import CrossEncoder
    return CrossEncoder(resolve_rerank_model_path(model_id), max_length=512)


_rerank_model_override: str | None = None
# 最近一次精排失败原因（供评测/健康检查读取，识别静默降级）
_rerank_last_error: dict = {"error": None}


def get_rerank_last_error() -> str | None:
    """返回最近一次 Rerank 失败原因；None 表示从未失败（或已成功覆盖）。"""
    return _rerank_last_error.get("error")


def set_rerank_model(model_name: str | None):
    """设置当前 rerank 模型（由主页面/API 启动时显式调用，避免图工作线程读 session_state 兜底）"""
    global _rerank_model_override
    _rerank_model_override = model_name


def rerank_docs(question: str, docs: list, top_n: int) -> list:
    """使用 Rerank 模型对候选文档按相关性精排（得分归一化为 0~1 置信度），返回前 top_n 个。
    注意：本函数会在图节点工作线程中被调用，不能直接依赖 st.session_state。"""
    if not docs:
        return docs
    rerank_choice = _rerank_model_override
    if not rerank_choice:
        try:
            rerank_choice = st.session_state.rerank_model
        except Exception:
            rerank_choice = "bge-reranker-base（更快，CPU 友好）"
    model_key = RERANK_MODEL_OPTIONS.get(rerank_choice, list(RERANK_MODEL_OPTIONS.keys())[0])
    try:
        reranker = load_reranker(model_key)
        pairs = [(question, d.page_content) for d in docs]
        scores = np.asarray(reranker.predict(pairs, show_progress_bar=False), dtype=float)
        _rerank_last_error["error"] = None  # 成功执行，清除历史失败标记
        if scores.size and (scores.max() > 1.0 or scores.min() < 0.0):
            scores = 1.0 / (1.0 + np.exp(-scores))
        ranked = sorted(zip(docs, scores), key=lambda x: float(x[1]), reverse=True)
        results = []
        for doc, score in ranked[:top_n]:
            doc.metadata["rerank_score"] = float(score)
            results.append(doc)
        return results
    except Exception as e:
        logger.error(f"Rerank 失败，降级为原始排序（前端若显示已启用精排则属异常）: {e}")
        _rerank_last_error["error"] = str(e)
        try:
            st.warning("精排服务暂时不可用，已使用原始排序结果。")
        except Exception:
            pass
        # 显式打标：调用方/评测必须能识别"请求了精排但实际未生效"，严禁把降级结果当成精排结果
        degraded = docs[:top_n]
        for d in degraded:
            d.metadata["_rerank_degraded"] = True
        return degraded


# =========================================================
# 文档格式化
# =========================================================
def format_docs(docs):
    """格式化检索到的文档"""
    if not docs:
        return "没有检索到相关资料。"
    result = []
    for idx, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source", "未知来源")
        chunk_id = doc.metadata.get("chunk_id", "未知块")
        result.append(
            f"【资料 {idx}】\n"
            f"来源文件：{source}\n"
            f"文档块编号：{chunk_id}\n"
            f"内容：{doc.page_content}"
        )
    return "\n\n".join(result)


# =========================================================
# 知识库一键重建（嵌入模型变更后使用）
# =========================================================
def rebuild_knowledge_base(
    vectorstore,
    chunk_size: int = 512,
    chunk_overlap: int = 100,
    progress_cb=None,
    strategy: str = "QA对切分",
) -> dict:
    """用当前 vectorstore 绑定的嵌入模型重建整个知识库。

    流程：清空向量集合 → 重读 docs/ 目录全部源文件 → 按 strategy 切分 → 重新入库
          → 递增 KB 版本（触发 BM25 索引重建）。
    strategy：切分方式（QA对切分 / 标题切分 / 语义切分），见 SPLIT_STRATEGIES。
    注意：不复用 add_file_to_vectorstore（其失败分支会删除源文件），重建
    过程任何单文件失败都只记录、绝不删除源文件。
    progress_cb: 可选回调 progress_cb(当前序号, 总数, 文件名)，用于 UI 进度条。
    返回 {"files": 文件数, "chunks": 总块数, "failed": [(文件名, 原因)]}
    """
    logger.info(f"知识库重建开始（清空旧向量 → 重读源文件 → {strategy} → 重新嵌入入库）")
    from datetime import datetime  # 与本模块其他入库函数保持一致的局部导入风格

    # 0. 预检嵌入服务：不可用时直接中止，绝不清空旧数据（防止中途失败导致知识库整体丢失）
    emb_fn = getattr(vectorstore, "_embedding_function", None)
    if emb_fn is not None:
        try:
            _probe = emb_fn.embed_query("知识库重建预检")
            if not _probe:
                raise RuntimeError("嵌入服务返回空向量")
        except Exception as e:
            logger.error(f"知识库重建中止：嵌入服务预检失败 {e}")
            return {"files": 0, "chunks": 0, "failed": [("嵌入服务预检失败",
                     f"嵌入服务不可用（{e}），已中止重建，旧知识库数据完好")]}

    # 1. 清空现有向量块
    existing_ids = vectorstore.get(include=[]).get("ids", [])
    if existing_ids:
        vectorstore.delete(ids=existing_ids)
        logger.info(f"已清空旧向量块 {len(existing_ids)} 个")

    # 2. 重读 docs/ 目录全部文件
    files = sorted(
        p for p in DOCS_DIR.iterdir()
        if p.is_file() and not p.name.startswith(".")
    )
    total_chunks = 0
    failed: list[tuple[str, str]] = []
    for i, path in enumerate(files, start=1):
        if progress_cb:
            try:
                progress_cb(i - 1, len(files), path.name)
            except Exception:
                pass
        try:
            chunks = split_file_for_preview(
                path, chunk_size, chunk_overlap, strategy,
                embeddings=getattr(vectorstore, "_embedding_function", None),
            )
            if not chunks:
                failed.append((path.name, "空文件或无法提取内容"))
                continue

            file_hash = calculate_file_hash(path.read_bytes())
            # source 展示名去掉入库时加的 12 位哈希前缀，与历史记录保持一致
            display_name = re.sub(r"^[0-9a-f]{12}_", "", path.name)
            uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            ids = []
            for idx, chunk in enumerate(chunks, start=1):
                chunk.metadata.update({
                    "source": display_name,
                    "file_hash": file_hash,
                    "chunk_id": idx,
                    "uploaded_at": uploaded_at,
                    "stored_path": str(path),
                })
                ids.append(f"{file_hash}_{idx}")
            vectorstore.add_documents(documents=chunks, ids=ids)
            total_chunks += len(chunks)
            logger.info(f"重建-文件入库成功 name={display_name} chunks={len(chunks)}")
        except Exception as e:
            logger.error(f"重建-文件处理失败 name={path.name}: {e}")
            failed.append((path.name, str(e)[:150]))

    # 3. 递增 KB 版本，触发 BM25 索引重建
    bump_kb_version()
    logger.info(
        f"知识库重建完成 files={len(files)} chunks={total_chunks} failed={len(failed)}"
    )
    return {"files": len(files), "chunks": total_chunks, "failed": failed}
