# -*- coding: utf-8 -*-
"""SQLite 会话持久化层（从 智能客服助手.py 拆出）"""
import json
import os
import sqlite3
import uuid
from datetime import datetime, timedelta, timezone
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
# 容器部署时通过 CHAT_DB_FILE 指向挂载卷内路径；本机默认项目根 chat_history.db
CHAT_DB_PATH = Path(os.environ.get("CHAT_DB_FILE") or (PROJECT_ROOT / "chat_history.db"))
CHAT_DB_PATH.parent.mkdir(parents=True, exist_ok=True)

# 统一时区：北京时间（UTC+8，无夏令时，固定偏移避免依赖 tzdata）。
# 时间写入与统计 cutoff 全部走 _now_str()，容器时区为 UTC 时每日统计不再差 8 小时。
_BJ_TZ = timezone(timedelta(hours=8))


def _now_str() -> str:
    """当前北京时间字符串（落库与统计统一口径）"""
    return datetime.now(_BJ_TZ).strftime("%Y-%m-%d %H:%M:%S")


def _init_chat_db() -> None:
    conn = sqlite3.connect(CHAT_DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute(
            """CREATE TABLE IF NOT EXISTS conversations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                title TEXT NOT NULL,
                token TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        conn.execute(
            """CREATE TABLE IF NOT EXISTS chat_messages (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                conv_id INTEGER NOT NULL,
                role TEXT NOT NULL,
                content TEXT NOT NULL,
                intent_name TEXT,
                route_note TEXT,
                rewritten_question TEXT,
                retrieval_mode TEXT,
                feedback TEXT,
                docs_json TEXT,
                created_at TEXT NOT NULL
            )"""
        )
        # 迁移：老库补 token 列（幂等，列已存在时跳过）
        try:
            conn.execute("ALTER TABLE conversations ADD COLUMN token TEXT")
        except sqlite3.OperationalError:
            pass
        # 迁移：为历史会话补随机 token（对外标识，防遍历）
        for (cid,) in conn.execute(
            "SELECT id FROM conversations WHERE token IS NULL OR token = ''"
        ).fetchall():
            conn.execute(
                "UPDATE conversations SET token = ? WHERE id = ?",
                (uuid.uuid4().hex, cid),
            )
        # 常用查询索引：会话内消息按时间、反馈筛选、报表按日期统计（幂等）
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv_time ON chat_messages(conv_id, created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_feedback ON chat_messages(feedback)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_messages_created ON chat_messages(created_at)"
        )
        conn.execute(
            "CREATE INDEX IF NOT EXISTS idx_conversations_created ON conversations(created_at)"
        )
        conn.commit()
    finally:
        conn.close()


_init_chat_db()


def db_execute(query: str, params: tuple = (), fetch: bool = False):
    """单次连接执行 SQL（自动提交/关闭，WAL 模式 + 30s 超时防并发锁）"""
    conn = sqlite3.connect(CHAT_DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        cur = conn.execute(query, params)
        if fetch:
            return cur.fetchall()
        conn.commit()
        return cur.lastrowid
    finally:
        conn.close()


def _conv_id_by_token(token: str) -> int | None:
    """根据对外 token 查内部自增 id；不存在返回 None。"""
    if not token:
        return None
    rows = db_execute(
        "SELECT id FROM conversations WHERE token = ?", (token,), fetch=True
    )
    return rows[0][0] if rows else None


def db_create_conversation(title: str) -> str:
    """创建会话，返回对外标识 token（随机 UUID，防遍历）。"""
    token = uuid.uuid4().hex
    db_execute(
        "INSERT INTO conversations (title, created_at, token) VALUES (?, ?, ?)",
        (title, _now_str(), token),
    )
    return token


def db_save_message(conv_token: str, msg: dict) -> int:
    conv_id = _conv_id_by_token(conv_token)
    if conv_id is None:
        raise ValueError(f"会话不存在：{conv_token}")
    docs_json = None
    if msg.get("docs"):
        docs_json = json.dumps(
            [
                {"page_content": d.page_content, "metadata": d.metadata}
                for d in msg["docs"]
            ],
            ensure_ascii=False,
        )
    return db_execute(
        """INSERT INTO chat_messages
           (conv_id, role, content, intent_name, route_note,
            rewritten_question, retrieval_mode, feedback, docs_json, created_at)
           VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
        (
            conv_id,
            msg.get("role", "assistant"),
            msg.get("content", ""),
            msg.get("intent_name"),
            msg.get("route_note"),
            msg.get("rewritten_question"),
            msg.get("retrieval_mode"),
            msg.get("feedback"),
            docs_json,
            datetime.now(_BJ_TZ).strftime("%Y-%m-%d %H:%M:%S"),
        ),
    )


def db_list_conversations(limit: int | None = None):
    """会话列表（默认全量，按 id 倒序）；返回 (token, title, created_at)。
    limit 传入时截取最近 N 条。"""
    query = "SELECT token, title, created_at FROM conversations ORDER BY id DESC"
    params: tuple = ()
    if limit is not None:
        query += " LIMIT ?"
        params = (limit,)
    return db_execute(query, params, fetch=True)


def db_load_messages(conv_token: str) -> list:
    """加载指定会话的全部消息（参考资料还原为 Document 对象）"""
    conv_id = _conv_id_by_token(conv_token)
    if conv_id is None:
        return []
    from langchain_core.documents import Document

    rows = db_execute(
        """SELECT role, content, intent_name, route_note, rewritten_question,
                  retrieval_mode, feedback, docs_json
           FROM chat_messages WHERE conv_id = ? ORDER BY id ASC""",
        (conv_id,),
        fetch=True,
    )
    messages = []
    for role, content, intent_name, route_note, rewritten_q, mode, feedback, docs_json in rows:
        msg = {"role": role, "content": content}
        if role == "assistant":
            if intent_name:
                msg["intent_name"] = intent_name
                msg["route_note"] = route_note or ""
            if rewritten_q:
                msg["rewritten_question"] = rewritten_q
            if mode:
                msg["retrieval_mode"] = mode
            if feedback:
                msg["feedback"] = feedback
            msg["docs"] = (
                [
                    Document(page_content=d["page_content"], metadata=d.get("metadata", {}))
                    for d in json.loads(docs_json)
                ]
                if docs_json
                else []
            )
        messages.append(msg)
    return messages


def db_update_feedback(msg_id: int, feedback: str) -> None:
    db_execute(
        "UPDATE chat_messages SET feedback = ? WHERE id = ?", (feedback, msg_id)
    )


def db_delete_message(msg_id: int) -> None:
    """删除单条消息（「重新生成」时清理旧回答，避免重开会话后旧回复复活）"""
    db_execute("DELETE FROM chat_messages WHERE id = ?", (msg_id,))


def db_delete_conversation(conv_token: str) -> None:
    """删除会话：级联删除消息表和会话表（单事务）"""
    conv_id = _conv_id_by_token(conv_token)
    if conv_id is None:
        return
    conn = sqlite3.connect(CHAT_DB_PATH, timeout=30)
    try:
        conn.execute("PRAGMA journal_mode=WAL")
        conn.execute("PRAGMA busy_timeout=30000")
        conn.execute("DELETE FROM chat_messages WHERE conv_id = ?", (conv_id,))
        conn.execute("DELETE FROM conversations WHERE id = ?", (conv_id,))
        conn.commit()
    finally:
        conn.close()


def db_export_all_messages() -> list:
    """导出全部会话的明细行（含会话与消息元数据），供 CSV 导出使用

    返回行: (会话ID, 会话标题, 会话创建时间, 角色, 消息内容,
             意图, 路由说明, 反馈, 消息时间)
    """
    return db_execute(
        """
        SELECT c.token, c.title, c.created_at,
               m.role, m.content, m.intent_name, m.route_note,
               m.feedback, m.created_at
        FROM conversations c
        LEFT JOIN chat_messages m ON m.conv_id = c.id
        ORDER BY c.id DESC, m.id ASC
        """,
        fetch=True,
    )


def _estimate_tokens(text: str) -> int:
    """粗略估算 token 数（中文≈1.5字/token，英文≈4字符/token，取混合近似）"""
    cn_chars = sum(1 for c in text if '\u4e00' <= c <= '\u9fff')
    other_chars = len(text) - cn_chars
    return int(cn_chars / 1.5 + other_chars / 4)


def build_history_text(
    messages: list,
    max_tokens: int = 1500,
    compress: bool = False,
    compress_threshold_tokens: int = 2000,
    summarizer=None,
) -> str:
    """将聊天记录转为文本，按 token 预算从最新消息向前截断（防止超窗口）。

    多轮上下文压缩（compress=True 时启用）：
        - 当总 token 超过 compress_threshold_tokens 时，把消息拆成「老段」+「近期 K 轮」；
        - 老段交给 summarizer（一般是 chat_model）压缩成 3-5 条要点；
        - 摘要 + 近期原文拼接后返回，token 数受 max_tokens 约束。
        - 适用场景：对话很长（>10 轮）时，避免老消息被截断丢弃导致改写/意图失忆。
    max_tokens 默认 1500，为 prompt 模板+检索 context+回答预留充足空间。
    至少保留最近一条消息，即使超出预算。
    """
    if not messages:
        return ""

    # 估算总 token：决定是否走压缩路径
    total_tokens = 0
    for msg in messages:
        c = msg.get("content", "")
        if c:
            total_tokens += _estimate_tokens(c)

    # 普通路径：按 max_tokens 从最新消息向前截断
    if not compress or total_tokens <= compress_threshold_tokens or summarizer is None:
        history_list = []
        token_sum = 0
        for i, msg in enumerate(reversed(messages)):
            role = msg["role"]
            content = msg.get("content", "")
            if not content:
                continue
            line = f"{'用户' if role == 'user' else '助手'}：{content}"
            line_tokens = _estimate_tokens(line)
            if token_sum + line_tokens > max_tokens and i > 0:
                break
            history_list.insert(0, line)
            token_sum += line_tokens
        return "\n".join(history_list)

    # 压缩路径：拆老段 / 近期，压缩老段
    # 近期：最近 4 轮（约 8 条消息）保留原文；其余视为「老段」压缩
    KEEP_RECENT_MESSAGES = 8
    recent = messages[-KEEP_RECENT_MESSAGES:]
    older = messages[:-KEEP_RECENT_MESSAGES]

    if not older:
        return build_history_text(
            messages, max_tokens=max_tokens, compress=False
        )

    older_text = "\n".join(
        f"{'用户' if m['role'] == 'user' else '助手'}：{m.get('content', '')}"
        for m in older if m.get("content")
    )
    try:
        summary = summarizer(older_text).strip()
        # 摘要超长时再做一次字符级截断（兜底，防止 summarizer 失控）
        if _estimate_tokens(summary) > max_tokens // 2:
            summary = summary[: max_tokens // 2]
    except Exception:
        # 压缩失败降级为不压缩、不丢历史
        logger.exception("历史压缩失败，降级为不压缩")
        return build_history_text(
            messages, max_tokens=max_tokens, compress=False
        )

    recent_text = build_history_text(
        recent, max_tokens=max_tokens - _estimate_tokens(summary) - 20,
        compress=False,
    )
    return (
        f"【历史摘要（共 {len(older)} 条早期对话已压缩为要点）】\n"
        f"{summary}\n\n"
        f"【近期对话（最近 {len(recent)} 条原文）】\n"
        f"{recent_text}"
    )


# =========================================================
# 运营报表统计（按客服运营日报口径聚合线上会话数据）
# =========================================================

def _cutoff(days: int | None) -> str | None:
    """统计起始时间（N 天前至今，北京时间口径，与落库 created_at 一致）；None 表示不限"""
    if not days:
        return None
    return (datetime.now(_BJ_TZ) - timedelta(days=days)).strftime("%Y-%m-%d %H:%M:%S")


def db_kpi_summary(days: int | None = None) -> dict:
    """运营 KPI 汇总：消息量、意图构成、兜底与反馈计数（口径见报表页说明）"""
    cutoff = _cutoff(days)
    where = "WHERE role = 'assistant'" + (" AND created_at >= ?" if cutoff else "")
    params = (cutoff,) if cutoff else ()
    row = db_execute(
        f"""
        SELECT COUNT(*),
               SUM(CASE WHEN intent_name = '知识库咨询' THEN 1 ELSE 0 END),
               SUM(CASE WHEN intent_name IN ('查订单', '创建工单') THEN 1 ELSE 0 END),
               SUM(CASE WHEN intent_name = '日常闲聊' THEN 1 ELSE 0 END),
               SUM(CASE WHEN route_note LIKE '%置信度兜底%' THEN 1 ELSE 0 END),
               SUM(CASE WHEN route_note LIKE '%无命中%' THEN 1 ELSE 0 END),
               SUM(CASE WHEN route_note LIKE '%检索异常%' THEN 1 ELSE 0 END),
               SUM(CASE WHEN route_note LIKE '%主动澄清%' THEN 1 ELSE 0 END),
               SUM(CASE WHEN feedback = 'up' THEN 1 ELSE 0 END),
               SUM(CASE WHEN feedback = 'down' THEN 1 ELSE 0 END)
        FROM chat_messages {where}
        """,
        params,
        fetch=True,
    )[0]
    keys = ["total", "kb", "tool", "chitchat",
            "fallback", "nohit", "retr_err", "clarify", "up", "down"]
    kpi = {k: (row[i] or 0) for i, k in enumerate(keys)}
    uwhere = "WHERE role = 'user'" + (" AND created_at >= ?" if cutoff else "")
    kpi["user"] = db_execute(
        f"SELECT COUNT(*) FROM chat_messages {uwhere}", params, fetch=True
    )[0][0]
    return kpi


def db_intent_distribution(days: int | None = None) -> list:
    """意图分布：[(意图, 次数)]，按次数降序"""
    cutoff = _cutoff(days)
    where = "WHERE role = 'assistant'" + (" AND created_at >= ?" if cutoff else "")
    params = (cutoff,) if cutoff else ()
    return db_execute(
        f"""SELECT COALESCE(intent_name, '未记录'), COUNT(*)
            FROM chat_messages {where}
            GROUP BY intent_name ORDER BY 2 DESC""",
        params,
        fetch=True,
    )


def db_daily_questions(days: int | None = None) -> list:
    """每日提问量趋势：[(日期, 提问数)]"""
    cutoff = _cutoff(days)
    where = "WHERE role = 'user'" + (" AND created_at >= ?" if cutoff else "")
    params = (cutoff,) if cutoff else ()
    return db_execute(
        f"""SELECT substr(created_at, 1, 10) AS d, COUNT(*)
            FROM chat_messages {where}
            GROUP BY d ORDER BY d""",
        params,
        fetch=True,
    )


def db_bad_cases(days: int | None = None) -> list:
    """Bad Case 明细：用户点踩 / 兜底 / 主动澄清 / 检索异常 / 检索无命中（关联出用户问题）

    返回行: (msg_id, conv_id, 时间, 意图, route_note, 反馈, 用户问题, 助手回答)
    """
    cutoff = _cutoff(days)
    where = (
        "a.role = 'assistant' AND (a.feedback = 'down'"
        " OR a.route_note LIKE '%兜底%'"
        " OR a.route_note LIKE '%主动澄清%'"
        " OR a.route_note LIKE '%检索异常%'"
        " OR a.route_note LIKE '%无命中%')"
    )
    params = []
    if cutoff:
        where += " AND a.created_at >= ?"
        params.append(cutoff)
    return db_execute(
        f"""
        SELECT a.id, a.conv_id, a.created_at, a.intent_name, a.route_note,
               COALESCE(a.feedback, ''),
               (SELECT u.content FROM chat_messages u
                 WHERE u.conv_id = a.conv_id AND u.role = 'user' AND u.id < a.id
                 ORDER BY u.id DESC LIMIT 1) AS question,
               a.content
        FROM chat_messages a
        WHERE {where}
        ORDER BY a.id DESC
        """,
        tuple(params),
        fetch=True,
    )
