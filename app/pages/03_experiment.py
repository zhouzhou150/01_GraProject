from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st

from asr_eval_system.service import run_default_experiment


st.title("测试任务")
st.write("可直接启动演示实验，系统会自动读取默认清单并完成评测、导出与入库。")

selected_models = st.multiselect(
    "选择模型",
    ["cnn_ctc", "rnn_ctc", "faster_whisper", "paddlespeech"],
    default=["cnn_ctc", "rnn_ctc", "faster_whisper", "paddlespeech"],
)

simulate = st.checkbox("使用模拟模式", value=True)

if st.button("开始评测", type="primary"):
    with st.spinner("正在运行评测任务，请稍候..."):
        report, exports = run_default_experiment(model_ids=selected_models, simulate=simulate)
    st.success(f"实验完成：{report.experiment_id}")
    st.dataframe(report.summary, use_container_width=True)
    st.json(exports)

