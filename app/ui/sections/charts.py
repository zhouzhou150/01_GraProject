from __future__ import annotations

import html

try:
    import altair as alt
except ImportError:  # pragma: no cover
    alt = None

import pandas as pd
import streamlit as st

from ui.constants import LOWER_IS_BETTER, METRIC_SPECS, MODEL_LIBRARY
from ui.helpers import section_header


def render_metric_chart(container, frame: pd.DataFrame, spec: dict[str, str]) -> None:
    metric_key, label, hint, fmt, color = spec["key"], spec["label"], spec["hint"], spec["fmt"], spec["color"]
    chart_data = frame[["model_label", metric_key]].copy().rename(columns={"model_label": "模型", metric_key: label})
    chart_data[label] = chart_data[label].fillna(0)
    chart_data = chart_data.sort_values(label, ascending=metric_key in LOWER_IS_BETTER)
    with container:
        st.markdown(f'<div class="chart-card"><h4 class="chart-title">{html.escape(label)}</h4><p class="chart-hint">{html.escape(hint)}</p></div>', unsafe_allow_html=True)
        if alt is None:  # pragma: no cover
            st.bar_chart(chart_data.set_index("模型"), width="stretch")
            return
        bar = (
            alt.Chart(chart_data)
            .mark_bar(cornerRadiusTopLeft=10, cornerRadiusTopRight=10, size=34)
            .encode(
                x=alt.X("模型:N", sort=list(chart_data["模型"]), axis=alt.Axis(labelAngle=0, title=None)),
                y=alt.Y(f"{label}:Q", title=None),
                color=alt.value(color),
                tooltip=[alt.Tooltip("模型:N"), alt.Tooltip(f"{label}:Q", format=fmt)],
            )
            .properties(height=250)
        )
        text = bar.mark_text(dy=-12, color="#2a211c", fontSize=12).encode(text=alt.Text(f"{label}:Q", format=fmt))
        st.altair_chart(bar + text, width="stretch")


def render_chart_section() -> None:
    section_header("05 / Charts", "可视化图表各模型对比", "为每个核心指标绘制单独图表，分别观察精度、性能和满意度维度的差异。")
    report = st.session_state["overall_report"]
    if report is None:
        st.info("完成总体测试后，这里会自动生成各指标对应的模型对比图表。")
        return
    frame = pd.DataFrame(report.summary).copy()
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    accuracy_specs = [spec for spec in METRIC_SPECS if spec["key"] in {"cer", "wer", "ser", "semdist"}]
    performance_specs = [spec for spec in METRIC_SPECS if spec["key"] in {"avg_latency_ms", "p95_latency_ms", "avg_upl_ms", "avg_rtf", "throughput", "load_time_ms", "cpu_pct", "mem_mb"}]
    satisfaction_specs = [spec for spec in METRIC_SPECS if spec["key"] in {"robustness_score", "resource_score", "uss"}]
    for tab, specs in zip(st.tabs(["精度与语义", "速度与资源", "满意度"]), [accuracy_specs, performance_specs, satisfaction_specs], strict=False):
        with tab:
            cols = st.columns(2, gap="large")
            for index, spec in enumerate(specs):
                render_metric_chart(cols[index % 2], frame, spec)
