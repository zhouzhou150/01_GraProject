from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

import streamlit as st

from asr_eval_system.service import database_path, list_saved_experiments, manifest_path, report_dir, runtime_dir


st.set_page_config(page_title="语音识别性能测度系统", page_icon=":bar_chart:", layout="wide")

st.title("深度学习语音识别性能测度系统")
st.write("这是一个面向本科毕业设计的本地评测系统，支持统一调用多种模型并输出综合指标。")

col1, col2, col3 = st.columns(3)
col1.metric("运行目录", str(runtime_dir()))
col2.metric("示例清单", str(manifest_path()))
col3.metric("报告目录", str(report_dir()))

st.subheader("当前能力")
st.markdown(
    """
- 支持 `CNN`、`RNN`、`faster-whisper`、`PaddleSpeech` 四类模型统一评测
- 支持 `CER/WER/SER/SemDist/RTF/UPL/USS`
- 支持结果持久化与 Markdown/CSV/JSON 报告导出
- 支持本地 Web 和桌面 GUI 共用一套后端
"""
)

st.subheader("历史实验")
experiments = list_saved_experiments()
if not experiments:
    st.info("暂无历史实验。请先运行 scripts/generate_demo_dataset.py 与 scripts/run_demo_experiment.py，或在“测试任务”页中直接启动。")
else:
    st.dataframe(experiments, use_container_width=True)

