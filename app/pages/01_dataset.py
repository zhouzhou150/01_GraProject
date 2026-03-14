from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st

from asr_eval_system.service import manifest_path, validate_default_manifest


st.title("数据集管理")
st.write("当前页面展示默认演示清单的内容与校验结果。")
st.code(str(manifest_path()))

try:
    items, issues = validate_default_manifest()
except FileNotFoundError:
    st.warning("尚未生成示例数据，请先运行 scripts/generate_demo_dataset.py。")
else:
    st.metric("样本数量", len(items))
    if issues:
        st.error("清单校验未通过")
        for issue in issues:
            st.write("- ", issue)
    else:
        st.success("清单校验通过")
    st.dataframe([item.to_dict() for item in items], use_container_width=True)

