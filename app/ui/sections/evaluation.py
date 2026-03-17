from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from ui.constants import MODEL_LIBRARY, SUMMARY_COLUMN_MAP
from ui.helpers import render_summary_cards, run_evaluation_workflow, section_header, summary_frame

LOGGER = logging.getLogger(__name__)


def _ranking_frame(frame: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    real_frame = frame[frame["runtime_mode"] == "真实"].copy()
    if not real_frame.empty:
        return real_frame, True
    return frame, False


def render_performance_results(report) -> None:
    frame = pd.DataFrame(report.summary).copy()
    if frame.empty:
        st.info("暂无性能测试结果。")
        return
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    ranking_source, filtered_to_real = _ranking_frame(frame)
    fastest = ranking_source.sort_values("avg_latency_ms", ascending=True).iloc[0]
    best_throughput = ranking_source.sort_values("throughput", ascending=False).iloc[0]
    shortest_load = ranking_source.sort_values("load_time_ms", ascending=True).iloc[0]
    render_summary_cards(
        [
            {
                "label": "最低平均延迟",
                "model": str(fastest["model_label"]),
                "value": f"{fastest['avg_latency_ms']:.2f} ms",
                "meta": f"{fastest['runtime_mode']} / {fastest['backend']}",
            },
            {
                "label": "最高吞吐量",
                "model": str(best_throughput["model_label"]),
                "value": f"{best_throughput['throughput']:.2f}",
                "meta": f"{best_throughput['runtime_mode']} / {best_throughput['backend']}",
            },
            {
                "label": "最快加载",
                "model": str(shortest_load["model_label"]),
                "value": f"{shortest_load['load_time_ms']:.2f} ms",
                "meta": f"{shortest_load['runtime_mode']} / {shortest_load['backend']}",
            },
        ]
    )
    if filtered_to_real:
        st.caption("摘要卡片已优先按真实模型排名；代理基线仍保留在下方表格中供对照。")
    table = frame[
        [
            "model_label",
            "runtime_mode",
            "backend",
            "load_time_ms",
            "avg_latency_ms",
            "p95_latency_ms",
            "avg_upl_ms",
            "avg_rtf",
            "throughput",
            "cpu_pct",
            "mem_mb",
        ]
    ]
    st.dataframe(table.rename(columns=SUMMARY_COLUMN_MAP), width="stretch", hide_index=True)


def _format_duration(seconds: float | None) -> str:
    if seconds is None or seconds < 0:
        return "计算中"
    total_seconds = int(round(seconds))
    minutes, secs = divmod(total_seconds, 60)
    hours, minutes = divmod(minutes, 60)
    if hours:
        return f"{hours:d}:{minutes:02d}:{secs:02d}"
    return f"{minutes:02d}:{secs:02d}"


def _stage_label(stage: str) -> str:
    return {
        "queued": "等待启动",
        "loading": "加载模型",
        "warmup": "预热模型",
        "running": "识别中",
        "finished": "当前模型完成",
        "complete": "测试完成",
        "error": "测试异常",
    }.get(stage, "处理中")


def _render_progress_snapshot(container, snapshot: dict[str, Any], started_at: float) -> None:
    fraction = 0.0
    if snapshot.get("overall_total_steps"):
        fraction = min(
            max(snapshot.get("overall_completed_steps", 0) / snapshot["overall_total_steps"], 0.0),
            1.0,
        )
    elapsed = time.perf_counter() - started_at
    estimated_total = (elapsed / fraction) if fraction > 0 else None
    remaining = (estimated_total - elapsed) if estimated_total is not None else None
    model_label = MODEL_LIBRARY.get(snapshot.get("model_id", ""), {}).get("label", snapshot.get("model_id", "等待中"))
    stage_text = _stage_label(str(snapshot.get("stage", "running")))
    sample_id = snapshot.get("sample_id") or "准备中"
    pred_text = snapshot.get("pred_text") or "等待模型返回识别文本..."
    ref_text = snapshot.get("ref_text") or "当前样本暂无参考文本。"
    backend = snapshot.get("backend") or "待初始化"
    runtime_mode = snapshot.get("runtime_mode") or "待确认"

    with container.container():
        st.progress(fraction)
        metric_cols = st.columns(4, gap="small")
        metric_cols[0].metric("已用时", _format_duration(elapsed))
        metric_cols[1].metric("预计总时长", _format_duration(estimated_total))
        metric_cols[2].metric("剩余时间", _format_duration(remaining))
        metric_cols[3].metric(
            "整体进度",
            f"{snapshot.get('overall_completed_steps', 0)}/{snapshot.get('overall_total_steps', 0)}",
        )
        st.caption(
            f"{stage_text} | 当前模型 {snapshot.get('model_index', 1)}/{snapshot.get('model_total', 1)}："
            f"{model_label} | 当前样本 {snapshot.get('sample_index', 0)}/{snapshot.get('sample_total', 0)}：{sample_id}"
        )
        st.caption(f"运行模式：{runtime_mode} | 后端：{backend}")
        text_cols = st.columns(2, gap="large")
        with text_cols[0]:
            st.markdown("**模型识别文本**")
            st.code(pred_text)
        with text_cols[1]:
            st.markdown("**参考文本**")
            st.code(ref_text)
        if snapshot.get("error_message"):
            st.error(str(snapshot["error_message"]))


def _progress_handler(container):
    started_at = time.perf_counter()

    def handle(snapshot: dict[str, Any]) -> None:
        _render_progress_snapshot(container, snapshot, started_at)

    return handle


def render_evaluation_section() -> None:
    section_header(
        "03 / Evaluation",
        "模型评估",
        "把性能测试和总体测试拆分开来：先做轻量基准，再对完整数据集运行总体评估、导出报告并写入历史记录。",
    )
    ready = bool(st.session_state["dataset_items"]) and bool(st.session_state["loaded_models"])
    tabs = st.tabs(["性能测试", "总体测试"])

    with tabs[0]:
        progress_panel = st.empty()
        if st.session_state["dataset_items"]:
            sample_limit = st.slider(
                "性能测试样本数",
                min_value=1,
                max_value=len(st.session_state["dataset_items"]),
                value=min(2, len(st.session_state["dataset_items"])),
            )
        else:
            sample_limit = 1
            st.caption("加载数据集后才能配置性能测试样本数。")
        if st.button("运行性能测试", key="run-performance", type="primary", disabled=not ready):
            try:
                report, _ = run_evaluation_workflow(
                    dataset_items=st.session_state["dataset_items"],
                    dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                    model_specs=list(st.session_state["loaded_models"].values()),
                    experiment_prefix="perf",
                    sample_limit=sample_limit,
                    export_bundle=False,
                    progress_callback=_progress_handler(progress_panel),
                )
            except Exception as exc:
                st.session_state["performance_report"] = None
                LOGGER.exception("Performance evaluation failed")
                st.error(f"性能测试失败：{type(exc).__name__}: {exc}")
            else:
                st.session_state["performance_report"] = report
                st.success("性能测试完成。")
        if st.session_state["performance_report"] is not None:
            render_performance_results(st.session_state["performance_report"])
        elif not ready:
            st.info("先完成数据导入和模型加载，性能测试按钮才会解锁。")

    with tabs[1]:
        progress_panel = st.empty()
        if st.session_state["dataset_items"]:
            overall_sample_limit = st.slider(
                "总体测试样本数",
                min_value=1,
                max_value=len(st.session_state["dataset_items"]),
                value=len(st.session_state["dataset_items"]),
                key="overall-sample-limit",
            )
            if overall_sample_limit < len(st.session_state["dataset_items"]):
                st.caption(f"当前仅对前 {overall_sample_limit} 条样本运行总体测试，适合大批量数据导入后的调试。")
            else:
                st.caption("当前会对数据集中的全部样本运行总体测试，并自动生成 JSON / CSV / Markdown 报告。")
        else:
            overall_sample_limit = 1
            st.caption("加载数据集后才能配置总体测试样本数。")
        if st.button("运行总体测试并导出", key="run-overall", type="primary", disabled=not ready):
            try:
                report, exports = run_evaluation_workflow(
                    dataset_items=st.session_state["dataset_items"],
                    dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                    model_specs=list(st.session_state["loaded_models"].values()),
                    experiment_prefix="eval",
                    sample_limit=overall_sample_limit,
                    export_bundle=True,
                    progress_callback=_progress_handler(progress_panel),
                )
            except Exception as exc:
                st.session_state["overall_report"] = None
                st.session_state["overall_exports"] = None
                LOGGER.exception("Overall evaluation failed")
                st.error(f"总体测试失败：{type(exc).__name__}: {exc}")
            else:
                st.session_state["overall_report"] = report
                st.session_state["overall_exports"] = exports
                st.success(f"总体测试完成，实验 ID：{report.experiment_id}")
        if st.session_state["overall_report"] is not None:
            st.dataframe(summary_frame(st.session_state["overall_report"].summary), width="stretch", hide_index=True)
            if st.session_state["overall_exports"]:
                exports = st.session_state["overall_exports"]
                st.caption(
                    f"已导出：JSON `{Path(exports['json']).name}` / "
                    f"CSV `{Path(exports['csv']).name}` / "
                    f"Markdown `{Path(exports['markdown']).name}`"
                )
        elif not ready:
            st.info("先完成数据导入和模型加载，随后即可运行完整实验。")
