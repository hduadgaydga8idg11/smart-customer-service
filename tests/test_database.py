# -*- coding: utf-8 -*-
"""database 模块单元测试（使用临时 SQLite，不污染真实数据）
运行：pytest tests/test_database.py -v
"""
import os
import tempfile
from pathlib import Path

import pytest

# 在 import core.database 之前替换 CHAT_DB_PATH 为临时文件
_tmp_dir = tempfile.mkdtemp()
_tmp_db = Path(_tmp_dir) / "test_chat.db"
os.environ["CHAT_DB_PATH"] = str(_tmp_db)

import core.database as database
database.CHAT_DB_PATH = _tmp_db
database._init_chat_db()

from core.database import (
    db_create_conversation,
    db_save_message,
    db_list_conversations,
    db_load_messages,
    db_update_feedback,
    db_delete_conversation,
    db_export_all_messages,
    build_history_text,
    db_execute,
)


class TestConversationCRUD:
    def test_create_and_list(self):
        conv_id = db_create_conversation("测试会话1")
        assert conv_id and len(conv_id) == 32  # 返回 32 位随机 token
        convs = db_list_conversations()
        assert any(c[1] == "测试会话1" for c in convs)

    def test_delete_conversation(self):
        conv_id = db_create_conversation("待删除会话")
        db_delete_conversation(conv_id)
        convs = db_list_conversations()
        assert not any(c[0] == conv_id for c in convs)

    def test_delete_cascades_messages(self):
        conv_id = db_create_conversation("级联删除测试")
        db_save_message(conv_id, {"role": "user", "content": "你好"})
        db_save_message(conv_id, {"role": "assistant", "content": "回复"})
        assert len(db_load_messages(conv_id)) == 2
        db_delete_conversation(conv_id)
        assert len(db_load_messages(conv_id)) == 0


class TestMessageSaveLoad:
    def test_save_and_load(self):
        conv_id = db_create_conversation("消息测试")
        db_save_message(conv_id, {"role": "user", "content": "你好"})
        db_save_message(conv_id, {"role": "assistant", "content": "你好，有什么可以帮你？"})
        msgs = db_load_messages(conv_id)
        assert len(msgs) == 2
        assert msgs[0]["role"] == "user"
        assert msgs[0]["content"] == "你好"
        assert msgs[1]["role"] == "assistant"
        assert msgs[1]["content"] == "你好，有什么可以帮你？"

    def test_save_with_metadata(self):
        conv_id = db_create_conversation("元数据测试")
        db_save_message(conv_id, {
            "role": "assistant",
            "content": "回复内容",
            "intent_name": "知识库咨询",
            "route_note": "RAG",
            "rewritten_question": "改写后问题",
            "retrieval_mode": "向量检索",
        })
        msgs = db_load_messages(conv_id)
        assert msgs[0]["intent_name"] == "知识库咨询"
        assert msgs[0]["rewritten_question"] == "改写后问题"
        assert msgs[0]["retrieval_mode"] == "向量检索"

    def test_save_with_docs(self):
        """测试保存带 Document 对象的消息"""
        from langchain_core.documents import Document
        conv_id = db_create_conversation("文档保存测试")
        docs = [Document(page_content="资料内容", metadata={"source": "test.md"})]
        db_save_message(conv_id, {"role": "assistant", "content": "回答", "docs": docs})
        msgs = db_load_messages(conv_id)
        assert len(msgs[0]["docs"]) == 1
        assert msgs[0]["docs"][0].page_content == "资料内容"

    def test_update_feedback(self):
        conv_id = db_create_conversation("反馈测试")
        msg_id = db_save_message(conv_id, {"role": "user", "content": "问题"})
        db_update_feedback(msg_id, "positive")
        # feedback 存储在 chat_messages 表，通过 db_execute 读取验证
        row = db_execute("SELECT feedback FROM chat_messages WHERE id = ?", (msg_id,), fetch=True)
        assert row[0][0] == "positive"


class TestBuildHistoryText:
    def test_empty(self):
        assert build_history_text([]) == ""

    def test_user_only(self):
        msgs = [{"role": "user", "content": "你好"}]
        result = build_history_text(msgs)
        assert "用户：你好" in result

    def test_assistant_only(self):
        msgs = [{"role": "assistant", "content": "回复"}]
        result = build_history_text(msgs)
        assert "助手：回复" in result

    def test_truncates_by_token_budget(self):
        """token 预算截断：超出 max_tokens 的旧消息被丢弃"""
        msgs = [{"role": "user", "content": f"问题{i}" * 200} for i in range(10)]
        result = build_history_text(msgs, max_tokens=100)
        assert len(result) > 0  # 至少保留最新的
        assert "问题9" in result  # 最新消息一定在
        assert "问题0" not in result  # 最早的被截断

    def test_mixed_roles(self):
        msgs = [
            {"role": "user", "content": "问题1"},
            {"role": "assistant", "content": "回答1"},
            {"role": "user", "content": "问题2"},
        ]
        result = build_history_text(msgs)
        assert "用户：问题1" in result
        assert "助手：回答1" in result
        assert "用户：问题2" in result


class TestFullListAndExport:
    def test_list_full_no_limit(self):
        """默认全量返回，不再截断最近 20 条"""
        created = [db_create_conversation(f"全量会话{i}") for i in range(3)]
        convs = db_list_conversations()
        got_ids = {c[0] for c in convs}
        for cid in created:
            assert cid in got_ids

    def test_list_with_limit(self):
        """显式传入 limit 时仅返回最近 N 条（兼容 API 分页）"""
        ids = [db_create_conversation(f"限量会话{i}") for i in range(4)]
        convs = db_list_conversations(limit=2)
        assert len(convs) == 2
        assert {c[0] for c in convs} == set(ids[2:])  # 倒序取最近两条

    def test_export_all_messages(self):
        """导出行包含会话与消息元数据，且同一会话内按消息先后排序"""
        conv_id = db_create_conversation("导出测试会话")
        db_save_message(conv_id, {"role": "user", "content": "导出问题"})
        db_save_message(conv_id, {
            "role": "assistant", "content": "导出回答",
            "intent_name": "知识库咨询", "route_note": "RAG 检索",
        })
        rows = db_export_all_messages()
        mine = [r for r in rows if r[0] == conv_id]
        assert len(mine) == 2
        # 列结构：(会话ID, 会话标题, 会话创建时间, 角色, 内容, 意图, 路由说明, 反馈, 消息时间)
        assert mine[0][1] == "导出测试会话"
        assert mine[0][3] == "user" and mine[0][4] == "导出问题"
        assert mine[1][3] == "assistant" and mine[1][4] == "导出回答"
        assert mine[1][5] == "知识库咨询" and mine[1][6] == "RAG 检索"

    def test_export_includes_empty_conversation(self):
        """无消息的空会话也应出现在导出中（LEFT JOIN 保留）"""
        conv_id = db_create_conversation("空会话导出")
        rows = db_export_all_messages()
        assert any(r[0] == conv_id and r[3] is None for r in rows)
