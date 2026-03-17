from __future__ import annotations

import html

import streamlit as st

from asr_eval_system.service import manifest_path
from asr_eval_system.workflow import WORKFLOW_STEP_TITLES, compute_workflow_progress
from ui.helpers import dataset_duration


def render_sidebar(history_count: int) -> None:
    dataset_ready = bool(st.session_state["dataset_items"])
    workflow = compute_workflow_progress(
        dataset_ready=dataset_ready,
        loaded_model_count=len(st.session_state["loaded_models"]),
        performance_ready=st.session_state["performance_report"] is not None,
        overall_ready=st.session_state["overall_report"] is not None,
    )
    with st.sidebar:
        st.markdown("### 评测流程")
        st.progress(workflow.progress_value)
        st.caption(f"当前建议继续：{workflow.current_step}")
        step_notes = (
            "加载示例数据，或上传音频和参考文本。",
            "确认模型后端与真实/模拟模式是否符合预期。",
            "先跑性能测试，再运行总体测试。",
            "检查结论、样本明细并导出报告。",
        )
        step_blocks = []
        for index, (title, done, note) in enumerate(zip(WORKFLOW_STEP_TITLES, workflow.completed_steps, step_notes, strict=False), start=1):
            done_class = " sidebar-step--done" if done else ""
            step_blocks.append(
                f'<div class="sidebar-step{done_class}"><span class="sidebar-step-index">{index}</span>'
                f'<span class="sidebar-step-copy">{html.escape(title)}<span class="sidebar-step-note">{html.escape(note)}</span></span></div>'
            )
        st.markdown(
            f"""
            <div class="sidebar-panel">
              <div class="sidebar-title">Recommended Flow</div>
              {''.join(step_blocks)}
            </div>
            """,
            unsafe_allow_html=True,
        )

        queue_text = "、".join(spec["label"] for spec in st.session_state["loaded_models"].values()) if st.session_state["loaded_models"] else "尚未加载模型"
        st.markdown(
            f"""
            <div class="sidebar-panel">
              <div class="sidebar-title">Current Status</div>
              <div class="sidebar-meta">数据源：{html.escape(st.session_state["dataset_label"])}</div>
              <div class="sidebar-meta">模型队列：{html.escape(queue_text)}</div>
              <div class="sidebar-meta">当前阶段：{html.escape(workflow.current_step)}</div>
              <div class="sidebar-meta">历史实验：{history_count} 次</div>
              <div class="sidebar-pill-row">
                <span class="sidebar-pill">样本 {len(st.session_state["dataset_items"])}</span>
                <span class="sidebar-pill">时长 {dataset_duration(st.session_state["dataset_items"]):.2f}s</span>
                <span class="sidebar-pill">模型 {len(st.session_state["loaded_models"])}</span>
              </div>
            </div>
            """,
            unsafe_allow_html=True,
        )


def render_hero(history_count: int) -> None:
    st.markdown(
        f"""
        <div class="hero">
          <div class="hero-kicker">Speech Evaluation Studio</div>
          <h1 class="hero-title">ASR 多模型评测工作台</h1>
          <p class="hero-copy">把语音导入、模型配置、性能测试、总体评估、结果导出和多指标图表整合成一个更清晰、更适合展示的单页界面。</p>
          <div class="tags">
            <span class="tag">数据源：{html.escape(st.session_state["dataset_label"])}</span>
            <span class="tag">示例清单：{html.escape(manifest_path().name)}</span>
            <span class="tag">历史实验：{history_count}</span>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
    cols = st.columns(3, gap="large")
    overview_cards = [
        ("当前样本数", str(len(st.session_state["dataset_items"])), "已进入当前评测工作流的数据条目"),
        ("数据总时长", f"{dataset_duration(st.session_state['dataset_items']):.2f} s", "用于性能测试和总体测试的音频总时长"),
        ("已加载模型", str(len(st.session_state["loaded_models"])), "已加入当前对比队列的模型数量"),
    ]
    for col, (label, value, meta) in zip(cols, overview_cards, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="overview-card">
                  <div class="overview-label">{html.escape(label)}</div>
                  <div class="overview-value">{html.escape(value)}</div>
                  <div class="overview-meta">{html.escape(meta)}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
