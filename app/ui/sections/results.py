from __future__ import annotations

import html
from pathlib import Path

import pandas as pd
import streamlit as st

from asr_eval_system.service import list_saved_experiments
from ui.constants import MODEL_LIBRARY
from ui.helpers import render_summary_cards, sample_frame, section_header


def _ranking_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    real_frame = frame[frame["runtime_mode"] == "真实"].copy()
    if not real_frame.empty:
        return real_frame, True
    return frame, False


def render_results_section() -> None:
    section_header(
        "04 / Insights",
        "结果分析与导出",
        "基于最近一次总体测试结果提炼关键结论，并提供下载按钮和历史实验记录，方便论文撰写与阶段汇报。",
    )
    report = st.session_state["overall_report"]
    exports = st.session_state["overall_exports"]
    if report is None:
        st.info("完成一次总体测试后，这里会自动出现结论摘要、导出按钮和逐样本结果。")
        return

    frame = pd.DataFrame(report.summary).copy()
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    ranking_source, filtered_to_real = _ranking_frame(frame)
    best_uss = ranking_source.sort_values("uss", ascending=False).iloc[0]
    best_cer = ranking_source.sort_values("cer", ascending=True).iloc[0]
    fastest = ranking_source.sort_values("avg_latency_ms", ascending=True).iloc[0]

    render_summary_cards(
        [
            {
                "label": "综合最佳",
                "model": str(best_uss["model_label"]),
                "value": f"{best_uss['uss']:.2f}",
                "meta": f"{best_uss['runtime_mode']} / {best_uss['backend']}",
            },
            {
                "label": "最低 CER",
                "model": str(best_cer["model_label"]),
                "value": f"{best_cer['cer']:.4f}",
                "meta": f"{best_cer['runtime_mode']} / {best_cer['backend']}",
            },
            {
                "label": "最快延迟",
                "model": str(fastest["model_label"]),
                "value": f"{fastest['avg_latency_ms']:.2f} ms",
                "meta": f"{fastest['runtime_mode']} / {fastest['backend']}",
            },
        ]
    )
    if filtered_to_real:
        st.caption("摘要卡片与实验结论已优先按真实模型排名；代理基线仍保留在明细表中。")

    ranking_hint = "在真实模型中，" if filtered_to_real else ""
    st.markdown(
        f"""
        <div class="insight">
          <h3>本轮实验结论</h3>
          <p>{ranking_hint}综合满意度最高的模型是 <strong>{html.escape(str(best_uss["model_label"]))}</strong>，USS 为 <strong>{best_uss["uss"]:.2f}</strong>。</p>
          <p>识别准确性最好的模型是 <strong>{html.escape(str(best_cer["model_label"]))}</strong>，CER 降至 <strong>{best_cer["cer"]:.4f}</strong>。</p>
          <p>平均延迟最快的模型是 <strong>{html.escape(str(fastest["model_label"]))}</strong>，平均延迟为 <strong>{fastest["avg_latency_ms"]:.2f} ms</strong>。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if exports:
        dcols = st.columns(3)
        dcols[0].download_button(
            "下载 JSON 报告",
            data=Path(exports["json"]).read_bytes(),
            file_name=Path(exports["json"]).name,
            mime="application/json",
            width="stretch",
        )
        dcols[1].download_button(
            "下载 CSV 摘要",
            data=Path(exports["csv"]).read_bytes(),
            file_name=Path(exports["csv"]).name,
            mime="text/csv",
            width="stretch",
        )
        dcols[2].download_button(
            "下载 Markdown 报告",
            data=Path(exports["markdown"]).read_bytes(),
            file_name=Path(exports["markdown"]).name,
            mime="text/markdown",
            width="stretch",
        )

    with st.expander("查看逐样本识别结果", expanded=False):
        st.dataframe(sample_frame(report.sample_results), width="stretch", hide_index=True)

    history = list_saved_experiments()
    if history:
        with st.expander("历史实验记录", expanded=False):
            st.dataframe(pd.DataFrame(history), width="stretch", hide_index=True)
