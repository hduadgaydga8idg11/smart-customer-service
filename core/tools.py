# -*- coding: utf-8 -*-
"""Agent 工具层：LangChain @tool 定义，供 Function Calling 使用

工具为纯 Python 实现（不依赖 UI / 会话状态，可在线程中安全调用）：
- 演示环境使用内存模拟数据
- 生产环境只需把函数体替换为真实业务系统 API 调用，
  LLM 侧 bind_tools 的参数 Schema 无需改动
"""
import uuid
from datetime import datetime
from typing import Literal

from langchain_core.tools import tool

# ---------- 模拟订单数据（生产环境替换为订单系统 API） ----------
MOCK_ORDERS = {
    "2024001": {
        "item": "无线蓝牙耳机",
        "status": "已发货",
        "carrier": "顺丰速运",
        "tracking_no": "SF1234567890",
        "eta": "预计明天下午送达",
    },
    "2024002": {
        "item": "机械键盘",
        "status": "运输中",
        "carrier": "中通快递",
        "tracking_no": "ZT9876543210",
        "eta": "预计 2 天后送达",
    },
    "2024003": {
        "item": "USB-C 数据线",
        "status": "已签收",
        "carrier": "京东物流",
        "tracking_no": "JD5555666677",
        "eta": "已送达",
    },
}


# ---------- 工具一：查询订单 ----------
def query_order_status(order_id: str) -> dict | None:
    """查询订单状态的底层实现（供节点直接调用；生产环境对接订单 API）"""
    return MOCK_ORDERS.get(order_id)


@tool
def query_order(order_id: str) -> dict:
    """根据订单号查询订单状态与物流信息。当用户想查询订单进度、物流配送情况时调用此工具。"""
    result = query_order_status(order_id)
    return dict(result) if result else {"error": f"未找到订单 {order_id} 的记录"}


# ---------- 工具二：创建工单 ----------
def create_ticket_record(issue: str, priority: str = "P2") -> dict:
    """创建工单的底层实现（供节点直接调用；生产环境对接工单系统 API）"""
    # 时间戳 + 4 位随机后缀：防止同一秒并发创建工单时撞号
    return {
        "ticket_id": f"GD{datetime.now().strftime('%Y%m%d%H%M%S')}{uuid.uuid4().hex[:4].upper()}",
        "issue": issue,
        "priority": priority,
        "created_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "sla": "24 小时内人工跟进" if priority == "P1" else "48 小时内人工跟进",
    }


@tool
def create_ticket(issue: str, priority: Literal["P1", "P2"] = "P2") -> dict:
    """为用户创建客服工单并登记问题描述。当用户投诉、报障、要求登记问题或人工处理时调用此工具。
    priority 取值：P1=紧急（涉及投诉、严重故障、强烈要求马上处理），P2=普通问题。"""
    return create_ticket_record(issue, priority=priority)


# 注册给 LLM bind_tools 的工具清单
ALL_TOOLS = [query_order, create_ticket]
