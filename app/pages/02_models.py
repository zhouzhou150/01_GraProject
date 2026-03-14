from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import streamlit as st

from asr_eval_system.models.registry import build_model_registry


st.title("模型管理")
st.write("展示系统当前支持的模型及其统一元数据。")

registry = build_model_registry(simulate=True)
rows = []
for model_id, adapter in registry.items():
    rows.append(
        {
            "model_id": model_id,
            "device": adapter.device,
            "backend": adapter.metadata()["backend"],
            "simulate": adapter.simulate,
        }
    )
st.dataframe(rows, use_container_width=True)

