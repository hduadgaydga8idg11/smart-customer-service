# =========================================================
# 知识库批量入库脚本（与 appv1.py 的 metadata schema 完全一致）
# 用法：.venv\Scripts\python.exe ingest_kb.py <文件路径> [更多文件路径...]
# =========================================================
import hashlib
import re
import shutil
import sys
from datetime import datetime
from pathlib import Path

from bs4 import BeautifulSoup
from langchain_chroma import Chroma
from langchain_community.document_loaders import TextLoader
from langchain_core.documents import Document
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

PROJECT_ROOT = Path(__file__).parent.parent.resolve()
DOCS_DIR = PROJECT_ROOT / "docs"
DB_PATH = str(PROJECT_ROOT / "chroma_db")
CHUNK_SIZE = 512
CHUNK_OVERLAP = 100

DOCS_DIR.mkdir(exist_ok=True)

# 与主程序 appv1 一致：文本类按纯文本读取，HTML 提取正文
SUPPORTED_SUFFIXES = {".txt", ".md", ".html", ".htm"}
HTML_SUFFIXES = {".html", ".htm"}


def load_documents(path: Path) -> list:
    """读取文件为 Document 列表；HTML 剥离脚本样式标签后提取正文，避免把源码切块入库"""
    if path.suffix.lower() in HTML_SUFFIXES:
        soup = BeautifulSoup(path.read_text(encoding="utf-8", errors="ignore"), "html.parser")
        for tag in soup(["script", "style", "noscript", "head", "meta", "link", "title"]):
            tag.decompose()
        lines = [ln.strip() for ln in soup.get_text("\n").splitlines()]
        text = "\n".join(ln for ln in lines if ln)
        if not text.strip():
            return []
        return [Document(page_content=text, metadata={"source": str(path)})]
    return TextLoader(str(path), encoding="utf-8").load()


def calculate_file_hash(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def is_file_already_in_db(vectorstore, file_hash: str) -> bool:
    records = vectorstore.get(where={"file_hash": file_hash}, limit=1)
    return bool(records["ids"])


def ingest_file(vectorstore, file_path: Path) -> None:
    # 若从 docs 目录的 hash 前缀副本入库，剥离前缀还原原始文件名
    clean_name = re.sub(r"^[0-9a-f]{64}_", "", file_path.name)
    raw = file_path.read_bytes()
    file_hash = calculate_file_hash(raw)
    if is_file_already_in_db(vectorstore, file_hash):
        print(f"⏭️ 已存在，跳过：{clean_name}")
        return

    if file_path.suffix.lower() not in SUPPORTED_SUFFIXES:
        print(f"❌ 暂不支持的格式：{clean_name}")
        return

    dest = DOCS_DIR / f"{file_hash}_{clean_name}"
    if dest.resolve() != file_path.resolve():
        shutil.copy2(file_path, dest)

    documents = load_documents(dest)
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
        separators=["\n\n", "\n", "。", "！", "？", "；", "，", " ", ""],
    )
    chunks = splitter.split_documents(documents)
    if not chunks:
        print(f"❌ 无可入库内容：{file_path.name}")
        return

    uploaded_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    ids = []
    for idx, chunk in enumerate(chunks, start=1):
        chunk.metadata.update(
            {
                "source": clean_name,
                "file_hash": file_hash,
                "chunk_id": idx,
                "uploaded_at": uploaded_at,
                "stored_path": str(dest),
            }
        )
        ids.append(f"{file_hash}_{idx}")

    vectorstore.add_documents(documents=chunks, ids=ids)
    print(f"✅ 入库成功：{clean_name} → {len(chunks)} 块（hash {file_hash[:12]}...）")


def main() -> None:
    if len(sys.argv) < 2:
        print("用法：python ingest_kb.py <文件路径> [更多文件路径...]")
        sys.exit(1)

    print("初始化向量库（Ollama bge-m3）...")
    embeddings = OllamaEmbeddings(
        model="bge-m3",
        client_kwargs={"timeout": 120},
    )
    vectorstore = Chroma(persist_directory=DB_PATH, embedding_function=embeddings)

    for arg in sys.argv[1:]:
        path = Path(arg)
        if not path.exists():
            print(f"❌ 文件不存在：{arg}")
            continue
        ingest_file(vectorstore, path)

    total = vectorstore._collection.count()
    print(f"\n完成。当前知识库总量：{total} 块")


if __name__ == "__main__":
    main()
