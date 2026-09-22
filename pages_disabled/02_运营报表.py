# =========================================================
# 运营报表（独立页）
#   线上真实会话的运营日报口径统计：KPI / 意图分布 / 每日提问量 /
#   知识咨询质量明细 / Bad Case 清单（支持 CSV 导出）。
#   新增：置信度兜底漏斗 + Bad Case 归因热力图。
# =========================================================

from datetime import datetime

import altair as alt
import pandas as pd
import streamlit as st

from core.database import (
    db_bad_cases, db_daily_questions, db_intent_distribution, db_kpi_summary,
)

st.set_page_config(page_title="运营报表", page_icon="📈", layout="wide")
st.title("📈 运营报表")

span = st.radio(
    "统计范围", ["最近 7 天", "最近 30 天", "全部"],
    horizontal=True, key="ops_span",
)
days = {"最近 7 天": 7, "最近 30 天": 30, "全部": None}[span]

kpi = db_kpi_summary(days)
if kpi["total"] == 0:
    st.warning("统计范围内暂无会话数据——先到主页对话几轮，再回来看报表。")
else:
    kb_total = kpi["kb"]
    fallback_total = kpi["fallback"] + kpi["nohit"] + kpi["retr_err"]
    fallback_rate = fallback_total / kb_total if kb_total else 0.0
    rated = kpi["up"] + kpi["down"]
    down_rate = kpi["down"] / rated if rated else 0.0

    # ---------- KPI ----------
    m1, m2, m3, m4, m5 = st.columns(5)
    m1.metric("👤 用户提问", f"{kpi['user']} 次")
    m2.metric("📚 知识咨询", f"{kb_total} 次",
              help=f"占回复的 {kb_total / kpi['total']:.0%}")
    m3.metric("🛠️ 工具调用", f"{kpi['tool']} 次",
              help=f"查订单 + 建工单；参数不足主动澄清 {kpi['clarify']} 次")
    m4.metric("🛟 兜底率", f"{fallback_rate:.0%}",
              help=f"触发兜底 {fallback_total} 次（置信度不足 {kpi['fallback']} / 无命中 {kpi['nohit']} / 异常 {kpi['retr_err']}）")
    m5.metric("👎 差评率", f"{down_rate:.0%}",
              help=f"👍 {kpi['up']} / 👎 {kpi['down']}（仅统计有点评的回复）")

    # ---------- 图表 ----------
    c1, c2 = st.columns(2)
    with c1:
        with st.container(border=True):
            st.markdown("##### 🎯 意图分布")
            intent_df = pd.DataFrame(
                db_intent_distribution(days), columns=["意图", "次数"]
            ).set_index("意图")
            if not intent_df.empty:
                st.bar_chart(intent_df)
    with c2:
        with st.container(border=True):
            st.markdown("##### 📈 每日提问量")
            daily_df = pd.DataFrame(
                db_daily_questions(days), columns=["日期", "提问量"]
            ).set_index("日期")
            if not daily_df.empty:
                st.bar_chart(daily_df)

    # ---------- 置信度兜底漏斗 ----------
    with st.container(border=True):
        st.markdown("##### 🛟 置信度兜底漏斗")
        st.caption(
            "从用户提问到最终给出可信回答，每一步过滤了多少请求。"
            "漏斗底端 = 真正由 RAG 生成并通过置信度判定的回答；"
            "漏斗上端 = 走到该层级的请求总数。"
        )
        user_total = kpi["user"]
        kb_total = kpi["kb"]
        fallback_total = kpi["fallback"] + kpi["nohit"] + kpi["retr_err"]
        retrieved = kb_total - kpi["nohit"] - kpi["retr_err"]
        confident = kb_total - fallback_total

        # 任一层为 0 时不画（避免漏斗断节）
        if user_total == 0 or kb_total == 0:
            st.info("暂无足够数据绘制漏斗（需要至少 1 条知识咨询）")
        else:
            stages = [
                ("👤 用户提问", user_total, kb_total,
                 f"{user_total - kb_total} 条分流到闲聊/工具"),
                ("📚 知识咨询", kb_total, retrieved,
                 f"{kpi['nohit'] + kpi['retr_err']} 条无召回被丢弃"),
                ("🔍 检索有命中", retrieved, confident,
                 f"{kpi['fallback']} 条置信度不足兜底"),
                ("✅ 置信度达标", confident, 0,
                 "成功作答"),
            ]
            funnel_df = pd.DataFrame({
                "阶段": [s[0] for s in stages],
                "请求数": [s[1] for s in stages],
                "流失说明": [s[3] for s in stages],
            })
            # 颜色由深到浅（漏斗形）；用 orangered 反向梯度让顶端最深、底端最浅
            chart = (
                alt.Chart(funnel_df)
                .mark_bar(size=42)
                .encode(
                    x=alt.X(
                        "请求数:Q",
                        title="请求数",
                        scale=alt.Scale(domain=[0, user_total * 1.05]),
                        axis=alt.Axis(format="d"),
                    ),
                    y=alt.Y("阶段:N", sort=list(reversed(funnel_df["阶段"].tolist())), title=None),
                    color=alt.Color(
                        "阶段:N",
                        sort=funnel_df["阶段"].tolist(),
                        scale=alt.Scale(scheme="orangered"),
                        legend=None,
                    ),
                    tooltip=[
                        alt.Tooltip("阶段:N"),
                        alt.Tooltip("请求数:Q"),
                        alt.Tooltip("流失说明:N"),
                    ],
                )
                .properties(height=240)
            )
            text = chart.mark_text(
                align="left", baseline="middle", dx=8, color="#333", fontSize=12
            ).encode(text=alt.Text("请求数:Q", format="d"))
            st.altair_chart(chart + text, use_container_width=True)

            # 漏斗下方补一行转化率速览
            conv_kb = kb_total / user_total if user_total else 0
            conv_ret = retrieved / kb_total if kb_total else 0
            conv_conf = confident / kb_total if kb_total else 0
            mc1, mc2, mc3 = st.columns(3)
            mc1.metric("提问 → 知识咨询", f"{conv_kb:.0%}")
            mc2.metric("咨询 → 检索命中", f"{conv_ret:.0%}",
                       help=f"未命中 + 异常合计 {kpi['nohit'] + kpi['retr_err']} 条")
            mc3.metric("咨询 → 置信度达标", f"{conv_conf:.0%}",
                       help=f"兜底 {kpi['fallback']} 条（精排最高分 < 阈值）")

    # ---------- 知识咨询质量 ----------
    if kb_total:
        with st.container(border=True):
            st.markdown("##### 🧭 知识咨询质量明细")
            kb_df = pd.DataFrame({
                "指标": ["检索命中", "置信度兜底", "无命中", "检索异常"],
                "次数": [kb_total - fallback_total, kpi["fallback"],
                         kpi["nohit"], kpi["retr_err"]],
            })
            kb_df["占比"] = kb_df["次数"].map(lambda x: f"{x / kb_total:.0%}")
            st.dataframe(kb_df, use_container_width=True, hide_index=True)

    # ---------- Bad Case ----------
    with st.container(border=True):
        st.markdown("##### 🚨 Bad Case 清单")
        cases = db_bad_cases(days)

        # 归因关键字 → 可读原因（与下方表格的 _reason 保持一致）
        _REASON_RULES = [
            ("用户点踩", lambda note, fb: fb == "down"),
            ("置信度不足", lambda note, fb: "置信度兜底" in note),
            ("无命中", lambda note, fb: "无命中" in note),
            ("检索异常", lambda note, fb: "检索异常" in note),
            ("参数不足", lambda note, fb: "主动澄清" in note),
        ]

        def _reason(note: str, fb: str) -> str:
            return "；".join(name for name, pred in _REASON_RULES if pred(note, fb)) or "其他"

        def _reason_list(note: str, fb: str) -> list[str]:
            return [name for name, pred in _REASON_RULES if pred(note, fb)]

        if not cases:
            st.success("✅ 统计范围内没有 Bad Case")
        else:
            # ---------- Bad Case 归因热力图 ----------
            # 一条 Bad Case 可能同时命中多种归因（例：既无命中又被点踩），每个归因各计 1
            heat_rows = []
            for c in cases:
                intent = c[3] or "未知"
                for reason in _reason_list(c[4] or "", c[5]):
                    heat_rows.append({"意图": intent, "归因": reason})
            heat_df = pd.DataFrame(heat_rows)
            if not heat_df.empty:
                matrix = heat_df.groupby(["意图", "归因"]).size().reset_index(name="次数")
                # 按出现频次高→低固定纵轴顺序，固定热力图配色梯度
                intent_order = list(matrix["意图"].value_counts().index)
                reason_order = [name for name, _ in _REASON_RULES]
                chart = (
                    alt.Chart(matrix)
                    .mark_rect()
                    .encode(
                        x=alt.X("归因:N", sort=reason_order, title="失败归因"),
                        y=alt.Y("意图:N", sort=intent_order, title="意图"),
                        color=alt.Color(
                            "次数:Q", scale=alt.Scale(scheme="orangered"), title="次数"
                        ),
                        tooltip=[
                            alt.Tooltip("意图:N"),
                            alt.Tooltip("归因:N"),
                            alt.Tooltip("次数:Q"),
                        ],
                    )
                    .properties(height=max(120, 40 * len(intent_order)))
                )
                st.caption(
                    f"📊 归因热力图：哪类意图在哪个环节最容易出问题（次数 = {len(cases)} 条 Bad Case 各自命中的归因合计）"
                )
                st.altair_chart(chart, use_container_width=True)

            case_df = pd.DataFrame(
                [(c[2], c[3] or "未知", (c[6] or "(未取到问题)"), _reason(c[4] or "", c[5]))
                 for c in cases],
                columns=["时间", "意图", "用户问题", "原因"],
            )
            st.dataframe(case_df, use_container_width=True, hide_index=True)
            with st.expander("查看逐条详情"):
                for c in cases:
                    q = c[6] or "(未取到问题)"
                    st.markdown(f"**{c[2]}｜{(c[3] or '未知')}｜{q[:40]}**")
                    st.markdown(f"用户问题：{q}")
                    st.markdown(f"助手回答：{c[7]}")
                    st.caption(f"路由：{c[4]}")
                    st.divider()
            st.download_button(
                "📥 导出 Bad Case (CSV)",
                data=case_df.to_csv(index=False).encode("utf-8-sig"),
                file_name=f"bad_cases_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )
