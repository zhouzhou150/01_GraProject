from __future__ import annotations

import pandas as pd
import streamlit as st

from ui.constants import MODEL_LIBRARY, SUMMARY_COLUMN_MAP
from ui.helpers import render_summary_cards, run_evaluation_workflow, section_header, summary_frame


def render_performance_results(report) -> None:
    frame = pd.DataFrame(report.summary).copy()
    if frame.empty:
        st.info("暂无性能测试结果。")
        return
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    fastest = frame.sort_values("avg_latency_ms", ascending=True).iloc[0]
    best_throughput = frame.sort_values("throughput", ascending=False).iloc[0]
    shortest_load = frame.sort_values("load_time_ms", ascending=True).iloc[0]
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
    table = frame[["model_label", "runtime_mode", "backend", "load_time_ms", "avg_latency_ms", "p95_latency_ms", "avg_upl_ms", "avg_rtf", "throughput", "cpu_pct", "mem_mb"]]
    st.dataframe(table.rename(columns=SUMMARY_COLUMN_MAP), width="stretch", hide_index=True)


def render_evaluation_section() -> None:
    section_header("03 / Evaluation", "模型评估", "把性能测试和总体测试拆分开来：先做轻量基准，再对完整数据集运行总体评估、导出报告并写入历史记录。")
    ready = bool(st.session_state["dataset_items"]) and bool(st.session_state["loaded_models"])
    tabs = st.tabs(["性能测试", "总体测试"])

    with tabs[0]:
        if st.session_state["dataset_items"]:
            sample_limit = st.slider("性能测试样本数", min_value=1, max_value=len(st.session_state["dataset_items"]), value=min(2, len(st.session_state["dataset_items"])))
        else:
            sample_limit = 1
            st.caption("加载数据集后才能配置性能测试样本数。")
        if st.button("运行性能测试", key="run-performance", type="primary", disabled=not ready):
            try:
                with st.spinner("正在进行性能测试，请稍候..."):
                    report, _ = run_evaluation_workflow(
                        dataset_items=st.session_state["dataset_items"],
                        dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                        model_specs=list(st.session_state["loaded_models"].values()),
                        experiment_prefix="perf",
                        sample_limit=sample_limit,
                        export_bundle=False,
                    )
            except Exception as exc:
                st.session_state["performance_report"] = None
                st.error(f"性能测试失败：{exc}")
            else:
                st.session_state["performance_report"] = report
                st.success("性能测试完成。")
        if st.session_state["performance_report"] is not None:
            render_performance_results(st.session_state["performance_report"])
        elif not ready:
            st.info("先完成数据导入和模型加载，性能测试按钮才会解锁。")

    with tabs[1]:
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
                with st.spinner("正在执行总体测试并生成导出文件，请稍候..."):
                    report, exports = run_evaluation_workflow(
                        dataset_items=st.session_state["dataset_items"],
                        dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                        model_specs=list(st.session_state["loaded_models"].values()),
                        experiment_prefix="eval",
                        sample_limit=overall_sample_limit,
                        export_bundle=True,
                    )
            except Exception as exc:
                st.session_state["overall_report"] = None
                st.session_state["overall_exports"] = None
                st.error(f"总体测试失败：{exc}")
            else:
                st.session_state["overall_report"] = report
                st.session_state["overall_exports"] = exports
                st.success(f"总体测试完成，实验 ID：{report.experiment_id}")
        if st.session_state["overall_report"] is not None:
            st.dataframe(summary_frame(st.session_state["overall_report"].summary), width="stretch", hide_index=True)
            if st.session_state["overall_exports"]:
                exports = st.session_state["overall_exports"]
                from pathlib import Path

                st.caption(f"已导出：JSON `{Path(exports['json']).name}` / CSV `{Path(exports['csv']).name}` / Markdown `{Path(exports['markdown']).name}`")
        elif not ready:
            st.info("先完成数据导入和模型加载，随后即可运行完整实验。")
