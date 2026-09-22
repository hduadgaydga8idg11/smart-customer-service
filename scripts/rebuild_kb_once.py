"""一次性脚本：空库上重建知识库（与 UI「重建知识库」走同一函数）。

前置：chroma_db 已初始化为空库（应用启动过一次），.env 内含嵌入服务 Key。
运行：.venv/Scripts/python scripts/rebuild_kb_once.py
"""
import importlib
import logging
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")

app = importlib.import_module("智能客服助手")  # 主入口提供 get_runtime
rt = app.get_runtime()

from core.retrieval import rebuild_knowledge_base  # noqa: E402


def cb(done: int, total: int) -> None:
    print(f"  入库进度 {done}/{total}", flush=True)


result = rebuild_knowledge_base(rt["vectorstore"], progress_cb=cb)
print("重建结果:", result)
sys.exit(0 if result.get("ok", False) else 1)
