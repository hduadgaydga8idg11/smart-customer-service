"""验证 + 截图一体脚本：先验证关键页面功能正常，再落 README 截图。

流程：
  1. 主对话页：发送知识库问题 → 等回答与参考资料 → 验证回答非空 → 截 main_chat.png
  2. 知识库管理页：验证无 InternalError/Traceback（Chroma 损坏回归检查）
  3. 模型设置页：切 API 模式 → 验证「接口地址 (Base URL)」输入框出现
  4. 评测系统页：整页截图 chain_eval.png

前置：Streamlit (8501) 已启动；本地 Ollama 可用。
运行：.venv/Scripts/python scripts/take_screenshots.py
"""
import sys
import time
from pathlib import Path

from playwright.sync_api import sync_playwright

BASE_URL = "http://localhost:8501"
ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = ROOT / "docs" / "screenshots"
VIEWPORT = {"width": 1440, "height": 900}
TEST_QUESTION = "耳机保修多久？"


def _wait_ready(page, timeout_ms: int = 30000) -> None:
    page.wait_for_selector("[data-testid='stSidebar']", timeout=timeout_ms)
    time.sleep(3.0)


def _assert_no_error(page, where: str) -> None:
    content = page.content()
    for bad in ("InternalError", "Traceback", "StreamlitAPIException"):
        if bad in content:
            raise AssertionError(f"[{where}] 页面出现异常标记: {bad}")


def _goto(page, nav_label: str) -> None:
    link = page.locator(
        f"[data-testid='stSidebarNav'] a:has-text('{nav_label}')"
    ).first
    link.wait_for(state="visible", timeout=15000)
    link.click()
    _wait_ready(page)
    time.sleep(2.0)


def _ask_and_wait(page) -> str:
    """主对话页发一条问题，返回回答文本"""
    print(f"→ [冒烟] 发送问题：{TEST_QUESTION}", flush=True)
    box = page.locator("[data-testid='stChatInput'] textarea").first
    box.wait_for(state="visible", timeout=20000)
    box.click()
    box.fill(TEST_QUESTION)
    box.press("Enter")
    page.wait_for_function(
        "document.querySelectorAll('[data-testid=\\'stChatMessage\\']').length >= 2",
        timeout=30000,
    )
    print("  等待回答生成…", flush=True)
    page.wait_for_function(
        """() => new Promise(resolve => {
            const sel = '[data-testid=\\'stChatMessage\\']';
            let last = document.querySelectorAll(sel).length +
                       document.body.innerText.length;
            let stable = 0;
            const timer = setInterval(() => {
                const cur = document.querySelectorAll(sel).length +
                            document.body.innerText.length;
                if (cur === last) { stable += 1; } else { stable = 0; }
                last = cur;
                if (stable >= 6) { clearInterval(timer); resolve(true); }
            }, 1000);
        })""",
        timeout=120000,
    )
    time.sleep(2.0)
    msgs = page.locator("[data-testid='stChatMessage']").all_inner_texts()
    answer = "\n".join(msgs[1:]) if len(msgs) > 1 else ""
    if len(answer.strip()) < 10:
        raise AssertionError("聊天冒烟失败：回答为空或过短")
    print(f"  ✓ 回答 {len(answer)} 字，冒烟通过", flush=True)
    return answer


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with sync_playwright() as p:
        browser = p.chromium.launch(headless=True, args=["--no-sandbox"])
        ctx = browser.new_context(viewport=VIEWPORT, locale="zh-CN")
        page = ctx.new_page()
        try:
            print(f"→ 访问 {BASE_URL}/", flush=True)
            page.goto(BASE_URL + "/", wait_until="domcontentloaded", timeout=60000)
            _wait_ready(page)

            # 1. 聊天冒烟 + 主对话页截图
            _ask_and_wait(page)
            _assert_no_error(page, "主对话")
            page.screenshot(path=str(OUT_DIR / "main_chat.png"))
            print(f"  ✓ {OUT_DIR / 'main_chat.png'}", flush=True)

            # 2. 知识库管理页：Chroma 损坏回归检查
            print("→ [验证] 知识库管理页", flush=True)
            _goto(page, "知识库管理")
            time.sleep(2.0)
            _assert_no_error(page, "知识库管理")
            page.screenshot(path=str(OUT_DIR / "kb_manage_check.png"))
            print("  ✓ 无报错，知识库页正常", flush=True)

            # 3. 模型设置页：验证 Base URL 输入框（切 API 模式）
            print("→ [验证] 模型设置页 Base URL 输入框", flush=True)
            _goto(page, "模型设置")
            chat_api = page.locator(
                "[data-testid='stRadio'] label:has-text('API')").first
            chat_api.wait_for(state="visible", timeout=15000)
            chat_api.click()
            time.sleep(2.5)  # 等 rerun
            n = page.locator("label:has-text('接口地址 (Base URL)')").count()
            if n < 1:
                raise AssertionError(f"Base URL 输入框未出现（匹配 {n} 个）")
            _assert_no_error(page, "模型设置")
            page.screenshot(path=str(OUT_DIR / "model_settings_check.png"))
            print(f"  ✓ Base URL 输入框已出现（{n} 处）", flush=True)

            # 4. 评测系统页整页截图
            print("→ [截图] 评测系统页", flush=True)
            _goto(page, "评测系统")
            _assert_no_error(page, "评测系统")
            page.screenshot(path=str(OUT_DIR / "chain_eval.png"), full_page=True)
            print(f"  ✓ {OUT_DIR / 'chain_eval.png'}", flush=True)
        finally:
            browser.close()
    print("全部验证通过，截图完成", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
