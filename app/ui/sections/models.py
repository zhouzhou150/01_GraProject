from __future__ import annotations

from datetime import datetime
import html
import time

import streamlit as st

from asr_eval_system.models.registry import build_model_adapter
from ui.constants import MODEL_LIBRARY
from ui.helpers import model_option_summary, section_header
from ui.state import reset_reports, set_flash_notice


PROXY_BASELINE_MODELS = {"cnn_ctc", "rnn_ctc"}


def _remove_loaded_model(model_id: str) -> None:
    adapter = st.session_state.get("loaded_adapters", {}).pop(model_id, None)
    if adapter is not None:
        try:
            adapter.unload()
        except Exception:
            pass

    removed = st.session_state["loaded_models"].pop(model_id, None)
    if removed is None:
        st.warning("该模型当前不在评测队列中。")
        return

    reset_reports()
    set_flash_notice("info", f"已移除 {removed['label']}。")
    st.rerun()


def render_model_cards() -> None:
    specs = list(st.session_state["loaded_models"].values())
    if not specs:
        st.info("尚未加载任何模型。先在左侧完成一次加载，右侧会自动形成对比队列。")
        return

    cols = st.columns(2, gap="large")
    for index, spec in enumerate(specs):
        with cols[index % 2]:
            title_col, action_col = st.columns([0.82, 0.18], gap="small")
            title_col.markdown(f'<div class="model-card-title">{html.escape(spec["label"])}</div>', unsafe_allow_html=True)
            with action_col:
                if st.button("×", key=f"remove-card-{spec['model_id']}", help=f"移除 {spec['label']}", width="content"):
                    _remove_loaded_model(spec["model_id"])

            st.markdown(
                f"""
                <div class="card">
                  <p>{html.escape(MODEL_LIBRARY[spec["model_id"]]["desc"])}</p>
                  <div>
                    <span class="chip">设备：{html.escape(spec["device"])}</span>
                    <span class="chip">请求模式：{html.escape(spec["requested_mode"])}</span>
                    <span class="chip">实际模式：{html.escape(spec["runtime_mode"])}</span>
                    <span class="chip">加载：{spec["load_time_ms"]:.2f} ms</span>
                  </div>
                  <p><strong>后端：</strong>{html.escape(spec["backend"])}</p>
                  <p><strong>后端详情：</strong>{html.escape(spec.get("backend_detail", "未提供"))}</p>
                  <p><strong>配置：</strong>{html.escape(model_option_summary(spec["options"]))}</p>
                  {f'<p class="muted">运行提示：{html.escape(spec["runtime_note"])}</p>' if spec.get("runtime_note") else ""}
                  <p class="muted">完成加载时间：{html.escape(spec["loaded_at"])}</p>
                  {f'<p class="muted">最近错误：{html.escape(spec["load_error"])}</p>' if spec.get("load_error") else ""}
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_section() -> None:
    section_header(
        "02 / Models",
        "模型选择与加载",
        "先选择模型，再设置运行配置，最后把模型加入当前评测队列。你可以重复这个流程来组建多模型对比集合。",
    )
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        model_id = st.selectbox("选择模型", options=list(MODEL_LIBRARY.keys()), format_func=lambda item: MODEL_LIBRARY[item]["label"])
        st.caption(MODEL_LIBRARY[model_id]["desc"])

        config_cols = st.columns(2, gap="large")
        with config_cols[0]:
            device = st.selectbox("部署设备", ["cpu", "cuda"], index=0)
            if model_id in PROXY_BASELINE_MODELS:
                st.toggle("使用模拟模式", value=True, disabled=True, key=f"simulate-proxy-{model_id}")
                simulate = True
                st.caption("该模型当前只提供代理基线，不参与真实模型加载。")
            else:
                simulate = st.toggle("使用模拟模式", value=True)
        with config_cols[1]:
            if model_id == "cnn_ctc":
                options = {
                    "decoder": st.selectbox("解码策略", ["greedy", "beam"]),
                    "frontend": st.selectbox("前端配置", ["light", "standard", "robust"], index=1),
                }
            elif model_id == "rnn_ctc":
                options = {
                    "hidden_size": st.selectbox("隐藏层规模", ["256", "384", "512"], index=1),
                    "dropout": st.selectbox("Dropout", ["0.1", "0.2", "0.3"], index=1),
                }
            elif model_id == "faster_whisper":
                compute_options = ["float16", "int8"] if device == "cuda" else ["int8", "float32"]
                options = {
                    "model_size": st.selectbox("Whisper 规模", ["tiny", "base", "small", "medium"], index=1),
                    "compute_type": st.selectbox("计算精度", compute_options, index=0),
                    "lang": st.selectbox("识别语言", ["zh", "en", "ja"], index=0),
                }
                if device == "cuda":
                    st.caption("CUDA 模式下默认使用更稳定的 float16 / int8，不再暴露 float32。")
            else:
                options = {
                    "lang": st.selectbox("语言", ["zh", "zh_en"], index=0),
                    "postprocess": st.selectbox("后处理", ["punctuation", "raw"], index=0),
                }

        st.caption("模型加入队列时会立即做一次真实加载校验；界面展示的是实际运行结果，而不是仅展示你的选择。")

        action_cols = st.columns(3, gap="small")
        if action_cols[0].button("加载模型到队列", type="primary", width="stretch"):
            progress = st.progress(0)
            status = st.empty()
            status.info("正在初始化模型配置...")
            progress.progress(0.2)
            time.sleep(0.08)

            adapter = build_model_adapter(model_id, device=device, simulate=simulate, options=options)
            status.info("正在执行加载校验...")
            progress.progress(0.65)
            try:
                adapter.load()
            except Exception as exc:
                progress.empty()
                status.empty()
                st.error(f"{MODEL_LIBRARY[model_id]['label']} 加载失败：{exc}")
            else:
                metadata = adapter.metadata()
                progress.progress(1.0)
                status.empty()
                progress.empty()
                st.session_state["loaded_models"][model_id] = {
                    "model_id": model_id,
                    "label": MODEL_LIBRARY[model_id]["label"],
                    "device": device,
                    "requested_mode": "模拟" if simulate else "真实",
                    "runtime_mode": "模拟" if bool(metadata.get("simulate", True)) else "真实",
                    "simulate": bool(metadata.get("simulate", True)),
                    "options": options,
                    "backend": str(metadata.get("backend", "unknown")),
                    "backend_detail": str(metadata.get("backend_detail", "")),
                    "runtime_note": str(metadata.get("runtime_note", "")),
                    "load_error": str(metadata.get("load_error", "")),
                    "load_time_ms": float(metadata.get("load_time_ms", 0.0)),
                    "loaded_at": datetime.now().strftime("%H:%M:%S"),
                }
                st.session_state.setdefault("loaded_adapters", {})[model_id] = adapter
                reset_reports()
                set_flash_notice("success", f"{MODEL_LIBRARY[model_id]['label']} 已加入当前评测队列。")
                st.rerun()

        if action_cols[1].button("移除当前模型", width="stretch"):
            _remove_loaded_model(model_id)

        if action_cols[2].button("清空模型队列", width="stretch"):
            for adapter in st.session_state.get("loaded_adapters", {}).values():
                try:
                    adapter.unload()
                except Exception:
                    pass
            st.session_state["loaded_models"] = {}
            st.session_state["loaded_adapters"] = {}
            reset_reports()
            set_flash_notice("info", "已清空当前模型队列。")
            st.rerun()

    with right:
        render_model_cards()
