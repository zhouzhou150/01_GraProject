from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st

from asr_eval_system.service import list_saved_experiments, load_saved_report


st.title("结果分析")
experiments = list_saved_experiments()
if not experiments:
    st.info("暂无可展示的实验结果。")
else:
    options = [item["experiment_id"] for item in experiments]
    selected = st.selectbox("选择实验", options)
    report = load_saved_report(selected)
    if report is not None:
        st.subheader("汇总指标")
        st.dataframe(report.summary, use_container_width=True)
        st.subheader("逐样本结果")
        st.dataframe(report.sample_results, use_container_width=True)
        st.subheader("满意度配置")
        st.json(report.satisfaction_profile)

