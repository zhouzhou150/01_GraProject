from __future__ import annotations

from datetime import datetime
import html
import json
from pathlib import Path
import sys
import time

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

try:
    import altair as alt
except ImportError:  # pragma: no cover
    alt = None

import pandas as pd
import streamlit as st

from asr_eval_system.data.audio_utils import read_wave_duration
from asr_eval_system.data.dataset import load_manifest, validate_manifest
from asr_eval_system.models.registry import build_model_adapter, build_model_registry_from_specs
from asr_eval_system.reporting.report_generator import export_report_bundle
from asr_eval_system.runner.evaluation import run_experiment
from asr_eval_system.schemas import DatasetManifest, ExperimentConfig
from asr_eval_system.service import (
    database_path,
    list_saved_experiments,
    load_default_profile,
    manifest_path,
    report_dir,
    runtime_dir,
)
from asr_eval_system.storage.database import DatabaseManager


st.set_page_config(page_title="ASR 多模型评测工作台", page_icon=":studio_microphone:", layout="wide")

MODEL_LIBRARY = {
    "cnn_ctc": {"label": "CNN-CTC", "desc": "轻量卷积基线，适合快速评测。", "accent": "#c96a2b"},
    "rnn_ctc": {"label": "RNN-CTC", "desc": "时序建模基线，强调稳定性。", "accent": "#6a8f7a"},
    "faster_whisper": {"label": "Faster-Whisper", "desc": "精度与速度平衡，可切换 Whisper 规模。", "accent": "#2f6c8f"},
    "paddlespeech": {"label": "PaddleSpeech", "desc": "中文语音生态友好，适合工程对比。", "accent": "#8f5a44"},
}

OPTION_LABELS = {
    "decoder": "解码策略",
    "frontend": "前端配置",
    "hidden_size": "隐藏层规模",
    "dropout": "Dropout",
    "model_size": "Whisper 规模",
    "compute_type": "计算精度",
    "lang": "语言",
    "postprocess": "后处理",
}

SUMMARY_COLUMN_MAP = {
    "model_label": "模型",
    "sample_count": "样本数",
    "cer": "CER",
    "wer": "WER",
    "ser": "SER",
    "semdist": "SemDist",
    "avg_latency_ms": "平均延迟(ms)",
    "p95_latency_ms": "P95 延迟(ms)",
    "avg_upl_ms": "UPL(ms)",
    "avg_rtf": "RTF",
    "throughput": "吞吐量",
    "cpu_pct": "CPU(%)",
    "mem_mb": "内存(MB)",
    "load_time_ms": "加载时间(ms)",
    "robustness_score": "鲁棒性得分",
    "resource_score": "资源效率得分",
    "uss": "USS",
    "satisfaction_level": "满意度等级",
}

SAMPLE_COLUMN_MAP = {
    "model_label": "模型",
    "sample_id": "样本 ID",
    "pred_text": "识别文本",
    "ref_text": "参考文本",
    "latency_ms": "延迟(ms)",
    "upl_ms": "UPL(ms)",
    "rtf": "RTF",
    "throughput": "吞吐量",
    "cpu_pct": "CPU(%)",
    "mem_mb": "内存(MB)",
    "load_time_ms": "加载时间(ms)",
    "cer": "CER",
    "wer": "WER",
    "ser": "SER",
    "semdist": "SemDist",
    "scene_tag": "场景",
    "noise_tag": "噪声",
    "status": "状态",
    "error_message": "错误信息",
}

METRIC_SPECS = [
    {"key": "cer", "label": "CER", "hint": "字符错误率，越低越好", "fmt": ".4f", "color": "#c96a2b"},
    {"key": "wer", "label": "WER", "hint": "词错误率，越低越好", "fmt": ".4f", "color": "#bc7a45"},
    {"key": "ser", "label": "SER", "hint": "句错误率，越低越好", "fmt": ".4f", "color": "#b35a42"},
    {"key": "semdist", "label": "SemDist", "hint": "语义距离，越高越好", "fmt": ".2f", "color": "#648d7c"},
    {"key": "avg_latency_ms", "label": "平均延迟(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#2f6c8f"},
    {"key": "p95_latency_ms", "label": "P95 延迟(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#497f9d"},
    {"key": "avg_upl_ms", "label": "UPL(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#7398ad"},
    {"key": "avg_rtf", "label": "RTF", "hint": "越低越好", "fmt": ".4f", "color": "#486d7d"},
    {"key": "throughput", "label": "吞吐量", "hint": "越高越好", "fmt": ".2f", "color": "#6279a0"},
    {"key": "load_time_ms", "label": "加载时间(ms)", "hint": "越低越好", "fmt": ".2f", "color": "#7d6748"},
    {"key": "cpu_pct", "label": "CPU(%)", "hint": "越低越好", "fmt": ".2f", "color": "#8e7a63"},
    {"key": "mem_mb", "label": "内存(MB)", "hint": "越低越好", "fmt": ".2f", "color": "#9a856d"},
    {"key": "robustness_score", "label": "鲁棒性得分", "hint": "越高越好", "fmt": ".2f", "color": "#6a8f7a"},
    {"key": "resource_score", "label": "资源效率得分", "hint": "越高越好", "fmt": ".2f", "color": "#758f68"},
    {"key": "uss", "label": "USS", "hint": "越高越好", "fmt": ".2f", "color": "#a35d32"},
]

LOWER_IS_BETTER = {"cer", "wer", "ser", "avg_latency_ms", "p95_latency_ms", "avg_upl_ms", "avg_rtf", "load_time_ms", "cpu_pct", "mem_mb"}

for key, value in {
    "dataset_items": [],
    "dataset_name": "",
    "dataset_label": "尚未加载数据集",
    "dataset_issues": [],
    "loaded_models": {},
    "performance_report": None,
    "overall_report": None,
    "overall_exports": None,
}.items():
    st.session_state.setdefault(key, value)


def reset_reports() -> None:
    st.session_state["performance_report"] = None
    st.session_state["overall_report"] = None
    st.session_state["overall_exports"] = None


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        :root {
          --page-bg: #f4eee4;
          --page-bg-deep: #ede1d0;
          --paper: rgba(255, 250, 243, 0.94);
          --paper-strong: rgba(255, 248, 238, 0.98);
          --ink: #231912;
          --ink-soft: #65584d;
          --ink-faint: #8c7765;
          --line: rgba(111, 86, 60, 0.16);
          --accent: #c96a2b;
          --accent-deep: #a84f20;
          --shadow: 0 20px 55px rgba(76, 54, 31, 0.10);
        }

        .stApp {
          background:
            radial-gradient(circle at top left, rgba(221, 170, 118, 0.20), transparent 30%),
            radial-gradient(circle at right 14%, rgba(86, 136, 164, 0.10), transparent 22%),
            linear-gradient(180deg, var(--page-bg) 0%, var(--page-bg-deep) 100%);
          color: var(--ink);
        }

        .block-container {
          max-width: 1280px;
          padding-top: 2.35rem;
          padding-bottom: 4rem;
        }

        [data-testid="stAppViewContainer"] .main,
        [data-testid="stAppViewContainer"] .main p,
        [data-testid="stAppViewContainer"] .main label,
        [data-testid="stAppViewContainer"] .main span,
        [data-testid="stAppViewContainer"] .main li,
        [data-testid="stAppViewContainer"] .main div {
          color: var(--ink);
        }

        h1, h2, h3 {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          color: var(--ink) !important;
          letter-spacing: -0.02em;
        }

        .hero {
          border: 1px solid var(--line);
          border-radius: 34px;
          padding: 2.6rem 2.4rem 2.2rem;
          background: linear-gradient(135deg, rgba(255, 249, 240, 0.96), rgba(244, 232, 215, 0.92));
          box-shadow: var(--shadow);
          margin-bottom: 1.35rem;
        }

        .hero-kicker, .tag, .chip {
          display: inline-flex;
          align-items: center;
          border-radius: 999px;
        }

        .hero-kicker {
          padding: .35rem .75rem;
          background: rgba(255, 255, 255, .72);
          border: 1px solid var(--line);
          font-size: .82rem;
          color: #8b5d3d !important;
          letter-spacing: .08em;
          text-transform: uppercase;
        }

        .hero-title {
          margin: .95rem 0 .55rem;
          font-size: clamp(2.1rem, 3.8vw, 3.5rem);
          line-height: 1.02;
          color: var(--ink) !important;
          font-weight: 700;
          max-width: 860px;
        }

        .hero-copy, .section-copy, .muted {
          color: var(--ink-soft) !important;
          line-height: 1.78;
        }

        .tags {
          display: flex;
          flex-wrap: wrap;
          gap: .65rem;
        }

        .tag {
          padding: .48rem .78rem;
          background: rgba(255, 251, 246, .92);
          border: 1px solid var(--line);
          color: #57493d !important;
          font-size: .9rem;
        }

        .section-kicker {
          color: #8b5b3d !important;
          font-size: .84rem;
          font-weight: 700;
          letter-spacing: .12em;
          text-transform: uppercase;
        }

        .section-title {
          margin: .1rem 0 0;
          font-size: 1.75rem;
          color: var(--ink) !important;
          font-weight: 700;
        }

        .card, .chart-card {
          border: 1px solid var(--line);
          box-shadow: var(--shadow);
        }

        .card {
          border-radius: 24px;
          background: var(--paper);
          padding: 1rem 1rem .95rem;
        }

        .chart-card {
          border-radius: 24px;
          background: var(--paper-strong);
          padding: .95rem 1rem .35rem;
          margin-bottom: 1rem;
        }

        .card h4, .chart-title {
          margin: 0 0 .45rem;
          color: var(--ink) !important;
          font-weight: 700;
        }

        .card p, .chart-hint {
          margin: .16rem 0;
          color: var(--ink-soft) !important;
          line-height: 1.65;
        }

        .chip {
          padding: .34rem .65rem;
          background: rgba(255, 255, 255, .86);
          border: 1px solid var(--line);
          color: #54483d !important;
          font-size: .82rem;
          margin: .25rem .25rem 0 0;
        }

        .insight {
          border: 1px solid var(--line);
          border-radius: 28px;
          padding: 1.3rem 1.4rem;
          background: linear-gradient(180deg, rgba(255, 249, 241, 0.98), rgba(249, 239, 226, 0.92));
          box-shadow: var(--shadow);
        }

        .insight h3, .insight p, .insight strong {
          color: var(--ink) !important;
        }

        .overview-card {
          border: 1px solid var(--line);
          border-radius: 24px;
          padding: 1rem 1rem 1.1rem;
          background: rgba(255, 249, 241, 0.95);
          box-shadow: var(--shadow);
          min-height: 118px;
        }

        .overview-label {
          font-size: 0.92rem;
          color: var(--ink-faint) !important;
          margin-bottom: 0.55rem;
          font-weight: 600;
        }

        .overview-value {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          font-size: clamp(2rem, 3vw, 2.5rem);
          line-height: 1;
          color: var(--ink) !important;
          font-weight: 700;
          margin-bottom: 0.35rem;
          word-break: break-word;
        }

        .overview-meta {
          color: var(--ink-soft) !important;
          font-size: 0.9rem;
          line-height: 1.55;
        }

        div[data-testid="stMetric"] {
          background: var(--paper-strong);
          border: 1px solid var(--line);
          padding: 1rem 1rem .9rem;
          border-radius: 20px;
          box-shadow: var(--shadow);
        }

        div[data-testid="stMetricLabel"],
        div[data-testid="stMetricLabel"] p,
        div[data-testid="stMetricValue"],
        div[data-testid="stMetricValue"] div,
        div[data-testid="stMetricDelta"],
        div[data-testid="stMetricDelta"] div {
          color: var(--ink) !important;
        }

        div[data-testid="stMetricValue"] {
          font-family: "Noto Serif SC", "Source Han Serif SC", Georgia, serif;
          font-weight: 700;
        }

        div[data-testid="stButton"] > button,
        div[data-testid="stDownloadButton"] > button {
          border-radius: 999px;
          border: 1px solid var(--line);
          background: linear-gradient(135deg, #fff6ec, #f4dfc6);
          color: #5a3f2a !important;
          font-weight: 700;
        }

        div[data-testid="stButton"] > button[kind="primary"] {
          background: linear-gradient(135deg, var(--accent), var(--accent-deep));
          color: #fff !important;
          border-color: transparent;
        }

        div[data-testid="stDataFrame"] {
          border-radius: 20px;
          overflow: hidden;
          border: 1px solid var(--line);
        }

        [data-testid="stSidebarNav"] {
          display: none;
        }

        [data-testid="stFileUploader"] label,
        [data-testid="stFileUploader"] small,
        [data-testid="stFileUploader"] span,
        [data-testid="stFileUploaderDropzoneInstructions"],
        [data-testid="stFileUploaderDropzoneInstructions"] * {
          color: var(--ink) !important;
        }

        [data-testid="stFileUploaderDropzone"] {
          background: rgba(252, 247, 239, 0.92);
          border: 1px dashed rgba(155, 121, 90, 0.55);
        }

        [data-testid="stFileUploader"] button,
        [data-testid="stFileUploaderDropzone"] button {
          background: rgba(255, 246, 235, 0.96) !important;
          color: #5a3f2a !important;
          border: 1px solid rgba(137, 101, 71, 0.22) !important;
          box-shadow: none !important;
        }

        button:disabled,
        button[disabled] {
          background: rgba(94, 79, 65, 0.12) !important;
          color: rgba(35, 25, 18, 0.52) !important;
          border-color: rgba(111, 86, 60, 0.14) !important;
          opacity: 1 !important;
        }

        [data-testid="stTextArea"] label,
        [data-testid="stSelectbox"] label,
        [data-testid="stMultiSelect"] label,
        [data-testid="stSlider"] label,
        [data-testid="stToggle"] label,
        [data-testid="stRadio"] label {
          color: var(--ink) !important;
          font-weight: 600;
        }

        [data-testid="stTextArea"] textarea,
        [data-baseweb="select"] > div,
        [data-testid="stNumberInput"] input {
          background: rgba(255, 250, 243, 0.96) !important;
          color: var(--ink) !important;
          border-color: rgba(140, 119, 101, 0.24) !important;
        }

        [data-testid="stTabs"] button {
          color: var(--ink) !important;
        }

        section[data-testid="stSidebar"] {
          background: linear-gradient(180deg, #262731 0%, #23242e 100%);
        }

        section[data-testid="stSidebar"] * {
          color: #f3ede8 !important;
        }

        .sidebar-panel {
          border: 1px solid rgba(255, 255, 255, 0.08);
          border-radius: 18px;
          padding: 0.95rem 0.95rem 0.85rem;
          background: rgba(255, 255, 255, 0.04);
          margin: 0.8rem 0 1rem;
        }

        .sidebar-title {
          font-size: 0.88rem;
          letter-spacing: 0.12em;
          text-transform: uppercase;
          color: #d8b28d !important;
          font-weight: 700;
          margin-bottom: 0.55rem;
        }

        .sidebar-step {
          display: flex;
          align-items: center;
          gap: 0.6rem;
          margin: 0.6rem 0;
          color: #f4eee9 !important;
        }

        .sidebar-step-index {
          width: 1.55rem;
          height: 1.55rem;
          border-radius: 999px;
          display: inline-flex;
          align-items: center;
          justify-content: center;
          background: rgba(216, 178, 141, 0.16);
          color: #f2d1b4 !important;
          font-size: 0.78rem;
          font-weight: 700;
        }

        .sidebar-meta {
          color: #d8d0c9 !important;
          font-size: 0.92rem;
          line-height: 1.65;
        }

        .sidebar-pill-row {
          display: flex;
          flex-wrap: wrap;
          gap: 0.45rem;
          margin-top: 0.75rem;
        }

        .sidebar-pill {
          display: inline-flex;
          align-items: center;
          padding: 0.35rem 0.65rem;
          border-radius: 999px;
          background: rgba(255, 255, 255, 0.08);
          border: 1px solid rgba(255, 255, 255, 0.08);
          color: #f6f0ea !important;
          font-size: 0.82rem;
        }

        section[data-testid="stSidebar"] .stProgress > div > div > div > div {
          background: linear-gradient(90deg, var(--accent), #e9ad76);
        }

        section[data-testid="stSidebar"] code {
          color: #ffd9b6 !important;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


def section_header(kicker: str, title: str, copy: str) -> None:
    st.markdown(
        f'<div class="section-kicker">{html.escape(kicker)}</div><h2 class="section-title">{html.escape(title)}</h2><p class="section-copy">{html.escape(copy)}</p>',
        unsafe_allow_html=True,
    )


def dataset_duration(items: list[DatasetManifest]) -> float:
    return round(sum(item.duration_sec for item in items), 2)


def dataset_preview_frame(items: list[DatasetManifest]) -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "样本 ID": item.sample_id,
                "文件名": Path(item.audio_path).name,
                "时长(s)": round(item.duration_sec, 2),
                "参考文本": item.transcript,
                "场景": item.scene_tag,
                "噪声": item.noise_tag,
            }
            for item in items
        ]
    )


def summary_frame(summary: list[dict]) -> pd.DataFrame:
    if not summary:
        return pd.DataFrame()
    frame = pd.DataFrame(summary).copy()
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    ordered = ["model_label"] + [key for key in SUMMARY_COLUMN_MAP if key in frame.columns and key != "model_label"]
    return frame[ordered].rename(columns=SUMMARY_COLUMN_MAP)


def sample_frame(sample_results: list[dict]) -> pd.DataFrame:
    if not sample_results:
        return pd.DataFrame()
    frame = pd.DataFrame(sample_results).copy()
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    ordered = ["model_label"] + [key for key in SAMPLE_COLUMN_MAP if key in frame.columns and key != "model_label"]
    return frame[ordered].rename(columns=SAMPLE_COLUMN_MAP)


def build_sidecar_transcript_map(uploaded_text_files: list | None) -> dict[str, str]:
    transcript_map: dict[str, str] = {}
    for transcript_file in uploaded_text_files or []:
        raw = transcript_file.getvalue()
        try:
            content = raw.decode("utf-8").strip()
        except UnicodeDecodeError:
            content = raw.decode("gbk", errors="ignore").strip()
        transcript_map[Path(transcript_file.name).stem] = content
    return transcript_map


def load_demo_dataset_with_progress() -> tuple[list[DatasetManifest], list[str]]:
    items = load_manifest(manifest_path())
    progress = st.progress(0)
    status = st.empty()
    total = max(len(items), 1)
    for index, item in enumerate(items, start=1):
        status.info(f"正在读取示例样本：{Path(item.audio_path).name}")
        time.sleep(0.08)
        progress.progress(index / total)
    issues = validate_manifest(items)
    progress.empty()
    status.empty()
    return items, issues


def save_uploaded_dataset(uploaded_audio_files: list, transcript_lookup: dict[str, str]) -> tuple[list[DatasetManifest], list[str], str]:
    batch_name = datetime.now().strftime("upload_%Y%m%d_%H%M%S")
    target_dir = runtime_dir() / "uploads" / batch_name
    target_dir.mkdir(parents=True, exist_ok=True)
    items: list[DatasetManifest] = []
    progress = st.progress(0)
    status = st.empty()
    total = max(len(uploaded_audio_files), 1)

    for index, uploaded_audio in enumerate(uploaded_audio_files, start=1):
        audio_name = Path(uploaded_audio.name).name
        audio_path = target_dir / audio_name
        if audio_path.exists():
            audio_path = target_dir / f"{audio_path.stem}_{index:02d}{audio_path.suffix}"
        status.info(f"正在导入：{audio_path.name}")
        audio_path.write_bytes(uploaded_audio.getvalue())
        ref_text = transcript_lookup.get(audio_name, "").strip()
        duration_sec = read_wave_duration(audio_path)
        if ref_text:
            audio_path.with_suffix(".txt").write_text(ref_text, encoding="utf-8")
        items.append(
            DatasetManifest(
                sample_id=f"upload_{index:02d}",
                audio_path=str(audio_path),
                transcript=ref_text,
                duration_sec=duration_sec,
                split="custom",
                scene_tag="uploaded",
                noise_tag="unknown",
                accent_tag="unknown",
            )
        )
        time.sleep(0.04)
        progress.progress(index / total)

    issues = validate_manifest(items)
    progress.empty()
    status.empty()
    return items, issues, batch_name


def model_option_summary(options: dict[str, str]) -> str:
    if not options:
        return "默认配置"
    return " / ".join(f"{OPTION_LABELS.get(key, key)}: {value}" for key, value in options.items())


def run_evaluation_workflow(
    dataset_items: list[DatasetManifest],
    dataset_name: str,
    model_specs: list[dict],
    experiment_prefix: str,
    sample_limit: int | None = None,
    export_bundle: bool = False,
) -> tuple[object, dict[str, str] | None]:
    selected_items = list(dataset_items[:sample_limit] if sample_limit else dataset_items)
    config = ExperimentConfig(
        experiment_id=datetime.now().strftime(f"{experiment_prefix}_%Y%m%d_%H%M%S"),
        model_ids=[spec["model_id"] for spec in model_specs],
        dataset_name=dataset_name,
        output_dir=str(report_dir()),
        notes=json.dumps(
            [
                {
                    "model_id": spec["model_id"],
                    "device": spec["device"],
                    "simulate": spec["simulate"],
                    "options": spec["options"],
                }
                for spec in model_specs
            ],
            ensure_ascii=False,
        ),
    )
    report = run_experiment(
        config=config,
        dataset_items=selected_items,
        model_registry=build_model_registry_from_specs(model_specs),
        profile=load_default_profile(),
    )
    exports = None
    if export_bundle:
        exports = export_report_bundle(report, report_dir())
        database = DatabaseManager(database_path())
        database.save_experiment(report)
        for export_type in ("json", "csv", "markdown"):
            database.record_export(report.experiment_id, export_type, exports[export_type], exports["created_at"])
    return report, exports


def render_sidebar(history_count: int) -> None:
    dataset_ready = bool(st.session_state["dataset_items"])
    model_ready = bool(st.session_state["loaded_models"])
    report_ready = st.session_state["overall_report"] is not None
    progress_value = (dataset_ready + model_ready + report_ready) / 3
    with st.sidebar:
        st.markdown("### 评测流程")
        st.progress(progress_value)
        st.caption("所有核心功能已经整合到首页，按下面的顺序完成一次完整评测即可。")
        st.markdown(
            """
            <div class="sidebar-panel">
              <div class="sidebar-title">Recommended Flow</div>
              <div class="sidebar-step"><span class="sidebar-step-index">1</span><span>导入音频与参考文本</span></div>
              <div class="sidebar-step"><span class="sidebar-step-index">2</span><span>选择模型并完成加载</span></div>
              <div class="sidebar-step"><span class="sidebar-step-index">3</span><span>先做性能测试，再跑总体测试</span></div>
              <div class="sidebar-step"><span class="sidebar-step-index">4</span><span>查看结论并导出报告</span></div>
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


def render_dataset_section() -> None:
    section_header("01 / Dataset", "导入音频文件", "支持一键加载示例音频，也支持上传自己的 WAV 文件并补充参考文本。加载完成后会展示文件名、时长和对应参考文本。")
    left, right = st.columns([1.15, 0.85], gap="large")

    with left:
        quick_col, clear_col = st.columns([0.7, 0.3], gap="small")
        if quick_col.button("加载示例数据", type="primary", use_container_width=True):
            items, issues = load_demo_dataset_with_progress()
            st.session_state["dataset_items"] = items if not issues else []
            st.session_state["dataset_name"] = "demo_manifest" if not issues else ""
            st.session_state["dataset_label"] = "示例数据集 / demo_manifest.json" if not issues else "示例数据集加载失败"
            st.session_state["dataset_issues"] = issues
            reset_reports()
            if issues:
                st.error("示例数据校验未通过：" + "；".join(issues))
            else:
                st.success("示例数据已加载，可以继续配置模型。")

        if clear_col.button("清空数据集", use_container_width=True):
            st.session_state["dataset_items"] = []
            st.session_state["dataset_name"] = ""
            st.session_state["dataset_label"] = "尚未加载数据集"
            st.session_state["dataset_issues"] = []
            reset_reports()
            st.info("已清空当前数据集。")

        submit_uploaded = False
        uploaded_audio_files = []
        with st.form("upload_dataset_form", clear_on_submit=False):
            uploaded_audio_files = st.file_uploader("上传 WAV 音频文件", type=["wav"], accept_multiple_files=True)
            uploaded_text_files = st.file_uploader("可选：上传同名 TXT 参考文本", type=["txt"], accept_multiple_files=True)
            transcript_defaults = build_sidecar_transcript_map(uploaded_text_files)
            if uploaded_audio_files:
                st.caption("请为每个音频填写参考文本，系统会在导入时进行有效性校验。")
                for audio_file in uploaded_audio_files:
                    st.text_area(
                        f"{audio_file.name} 的参考文本",
                        key=f"transcript::{audio_file.name}",
                        value=transcript_defaults.get(Path(audio_file.name).stem, ""),
                        height=78,
                    )
            else:
                st.caption("先选择一个或多个 WAV 文件，再执行导入。")
            submit_uploaded = st.form_submit_button("导入上传音频", use_container_width=True, disabled=not bool(uploaded_audio_files))

        if submit_uploaded:
            if not uploaded_audio_files:
                st.warning("请先选择至少一个 WAV 文件。")
            else:
                transcript_lookup = {
                    Path(audio_file.name).name: st.session_state.get(f"transcript::{audio_file.name}", "").strip()
                    for audio_file in uploaded_audio_files
                }
                items, issues, batch_name = save_uploaded_dataset(uploaded_audio_files, transcript_lookup)
                st.session_state["dataset_items"] = items if not issues else []
                st.session_state["dataset_name"] = batch_name if not issues else ""
                st.session_state["dataset_label"] = f"上传数据集 / {batch_name}" if not issues else "上传数据集校验失败"
                st.session_state["dataset_issues"] = issues
                reset_reports()
                if issues:
                    st.error("上传数据校验未通过：" + "；".join(issues))
                else:
                    st.success(f"已导入 {len(items)} 个音频样本。")

    with right:
        issues = st.session_state["dataset_issues"]
        st.markdown(
            f"""
            <div class="card">
              <h4>数据集概览</h4>
              <p><strong>来源：</strong>{html.escape(st.session_state["dataset_label"])}</p>
              <p><strong>样本数量：</strong>{len(st.session_state["dataset_items"])}</p>
              <p><strong>总时长：</strong>{dataset_duration(st.session_state["dataset_items"]):.2f} 秒</p>
              <p><strong>校验状态：</strong>{'通过' if st.session_state["dataset_items"] and not issues else '待处理'}</p>
              <p class="muted">如果要跑完整实验，建议先确保所有参考文本都已填写完整。</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if st.session_state["dataset_items"]:
        st.dataframe(dataset_preview_frame(st.session_state["dataset_items"]), use_container_width=True, hide_index=True)
        with st.expander("试听与文本预览", expanded=False):
            for item in st.session_state["dataset_items"][:3]:
                audio_col, text_col = st.columns([0.7, 1.3], gap="large")
                with audio_col:
                    st.audio(Path(item.audio_path).read_bytes(), format="audio/wav")
                with text_col:
                    st.markdown(f"**文件名**：`{Path(item.audio_path).name}`")
                    st.markdown(f"**参考文本**：{item.transcript}")
                    st.caption(f"场景：{item.scene_tag}  |  噪声：{item.noise_tag}  |  时长：{item.duration_sec:.2f}s")
    else:
        st.info("当前还没有可用于评测的音频数据。可以先加载示例数据，或者上传自己的 WAV 样本。")


def render_model_cards() -> None:
    specs = list(st.session_state["loaded_models"].values())
    if not specs:
        st.info("尚未加载任何模型。先在左侧选择模型并完成加载，右侧会自动形成对比队列。")
        return

    cols = st.columns(2, gap="large")
    for index, spec in enumerate(specs):
        with cols[index % 2]:
            st.markdown(
                f"""
                <div class="card">
                  <h4>{html.escape(spec["label"])}</h4>
                  <p>{html.escape(MODEL_LIBRARY[spec["model_id"]]["desc"])}</p>
                  <div>
                    <span class="chip">设备：{html.escape(spec["device"])}</span>
                    <span class="chip">模式：{'模拟' if spec['simulate'] else '真实'}</span>
                    <span class="chip">加载：{spec['load_time_ms']:.2f} ms</span>
                  </div>
                  <p class="muted">后端：{html.escape(spec["backend"])} / 加载时间：{html.escape(spec["loaded_at"])}</p>
                  <p class="muted">配置：{html.escape(model_option_summary(spec["options"]))}</p>
                </div>
                """,
                unsafe_allow_html=True,
            )


def render_model_section() -> None:
    section_header("02 / Models", "模型选择与加载", "先选择模型，再设置运行配置，最后把模型加入当前评测队列。你可以重复这个流程来组建多模型对比集合。")
    left, right = st.columns([0.95, 1.05], gap="large")

    with left:
        model_id = st.selectbox("选择模型", options=list(MODEL_LIBRARY.keys()), format_func=lambda item: MODEL_LIBRARY[item]["label"])
        st.caption(MODEL_LIBRARY[model_id]["desc"])
        config_cols = st.columns(2, gap="large")
        with config_cols[0]:
            device = st.selectbox("部署设备", ["cpu", "cuda"], index=0)
            simulate = st.toggle("使用模拟模式", value=True)
        with config_cols[1]:
            if model_id == "cnn_ctc":
                options = {"decoder": st.selectbox("解码策略", ["greedy", "beam"]), "frontend": st.selectbox("前端配置", ["light", "standard", "robust"], index=1)}
            elif model_id == "rnn_ctc":
                options = {"hidden_size": st.selectbox("隐藏层规模", ["256", "384", "512"], index=1), "dropout": st.selectbox("Dropout", ["0.1", "0.2", "0.3"], index=1)}
            elif model_id == "faster_whisper":
                options = {"model_size": st.selectbox("Whisper 规模", ["tiny", "base", "small", "medium"], index=1), "compute_type": st.selectbox("计算精度", ["int8", "float16", "float32"], index=0)}
            else:
                options = {"lang": st.selectbox("语言", ["zh", "zh_en"], index=0), "postprocess": st.selectbox("后处理", ["punctuation", "raw"], index=0)}
        st.caption("当前版本中，设备、模拟模式和 Whisper 规模会参与实际加载；其余配置会写入实验备注，便于保留对比上下文。")

        action_cols = st.columns(3, gap="small")
        if action_cols[0].button("加载模型到队列", type="primary", use_container_width=True):
            progress = st.progress(0)
            status = st.empty()
            status.info("正在初始化模型配置...")
            progress.progress(0.2)
            time.sleep(0.08)
            adapter = build_model_adapter(model_id, device=device, simulate=simulate, options=options)
            status.info("正在执行加载校验...")
            progress.progress(0.65)
            adapter.load()
            metadata = adapter.metadata()
            progress.progress(1.0)
            status.empty()
            progress.empty()
            st.session_state["loaded_models"][model_id] = {
                "model_id": model_id,
                "label": MODEL_LIBRARY[model_id]["label"],
                "device": device,
                "simulate": simulate,
                "options": options,
                "backend": str(metadata.get("backend", "unknown")),
                "load_time_ms": float(metadata.get("load_time_ms", 0.0)),
                "loaded_at": datetime.now().strftime("%H:%M:%S"),
            }
            reset_reports()
            st.success(f"{MODEL_LIBRARY[model_id]['label']} 已加入当前评测队列。")

        if action_cols[1].button("移除当前模型", use_container_width=True):
            if model_id in st.session_state["loaded_models"]:
                st.session_state["loaded_models"].pop(model_id)
                reset_reports()
                st.info(f"已移除 {MODEL_LIBRARY[model_id]['label']}。")
            else:
                st.warning("当前模型还没有加入评测队列。")

        if action_cols[2].button("清空模型队列", use_container_width=True):
            st.session_state["loaded_models"] = {}
            reset_reports()
            st.info("已清空当前模型队列。")

    with right:
        render_model_cards()


def render_metric_chart(container, frame: pd.DataFrame, spec: dict[str, str]) -> None:
    metric_key, label, hint, fmt, color = spec["key"], spec["label"], spec["hint"], spec["fmt"], spec["color"]
    chart_data = frame[["model_label", metric_key]].copy().rename(columns={"model_label": "模型", metric_key: label})
    chart_data[label] = chart_data[label].fillna(0)
    chart_data = chart_data.sort_values(label, ascending=metric_key in LOWER_IS_BETTER)
    with container:
        st.markdown(f'<div class="chart-card"><h4 class="chart-title">{html.escape(label)}</h4><p class="chart-hint">{html.escape(hint)}</p></div>', unsafe_allow_html=True)
        if alt is None:  # pragma: no cover
            st.bar_chart(chart_data.set_index("模型"), use_container_width=True)
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
        st.altair_chart(bar + text, use_container_width=True)


def render_performance_results(report) -> None:
    frame = pd.DataFrame(report.summary).copy()
    if frame.empty:
        st.info("暂无性能测试结果。")
        return
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    fastest = frame.sort_values("avg_latency_ms", ascending=True).iloc[0]
    best_throughput = frame.sort_values("throughput", ascending=False).iloc[0]
    shortest_load = frame.sort_values("load_time_ms", ascending=True).iloc[0]
    cols = st.columns(3)
    cols[0].metric("最低平均延迟", f"{fastest['model_label']} / {fastest['avg_latency_ms']:.2f} ms")
    cols[1].metric("最高吞吐量", f"{best_throughput['model_label']} / {best_throughput['throughput']:.2f}")
    cols[2].metric("最快加载", f"{shortest_load['model_label']} / {shortest_load['load_time_ms']:.2f} ms")
    table = frame[["model_label", "load_time_ms", "avg_latency_ms", "p95_latency_ms", "avg_upl_ms", "avg_rtf", "throughput", "cpu_pct", "mem_mb"]]
    st.dataframe(table.rename(columns=SUMMARY_COLUMN_MAP), use_container_width=True, hide_index=True)


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
            with st.spinner("正在进行性能测试，请稍候..."):
                report, _ = run_evaluation_workflow(
                    dataset_items=st.session_state["dataset_items"],
                    dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                    model_specs=list(st.session_state["loaded_models"].values()),
                    experiment_prefix="perf",
                    sample_limit=sample_limit,
                    export_bundle=False,
                )
            st.session_state["performance_report"] = report
            st.success("性能测试完成。")
        if st.session_state["performance_report"] is not None:
            render_performance_results(st.session_state["performance_report"])
        elif not ready:
            st.info("先完成数据导入和模型加载，性能测试按钮才会解锁。")

    with tabs[1]:
        st.caption("总体测试会对当前数据集中的全部样本运行完整评估，并自动生成 JSON / CSV / Markdown 报告。")
        if st.button("运行总体测试并导出", key="run-overall", type="primary", disabled=not ready):
            with st.spinner("正在执行总体测试并生成导出文件，请稍候..."):
                report, exports = run_evaluation_workflow(
                    dataset_items=st.session_state["dataset_items"],
                    dataset_name=st.session_state["dataset_name"] or "temporary_dataset",
                    model_specs=list(st.session_state["loaded_models"].values()),
                    experiment_prefix="eval",
                    export_bundle=True,
                )
            st.session_state["overall_report"] = report
            st.session_state["overall_exports"] = exports
            st.success(f"总体测试完成，实验 ID：{report.experiment_id}")
        if st.session_state["overall_report"] is not None:
            st.dataframe(summary_frame(st.session_state["overall_report"].summary), use_container_width=True, hide_index=True)
            if st.session_state["overall_exports"]:
                exports = st.session_state["overall_exports"]
                st.caption(f"已导出：JSON `{Path(exports['json']).name}` / CSV `{Path(exports['csv']).name}` / Markdown `{Path(exports['markdown']).name}`")
        elif not ready:
            st.info("先完成数据导入和模型加载，随后即可运行完整实验。")


def render_results_section() -> None:
    section_header("04 / Insights", "结果分析与导出", "基于最近一次总体测试结果提炼关键结论，并提供下载按钮和历史实验记录，方便论文撰写与阶段汇报。")
    report = st.session_state["overall_report"]
    exports = st.session_state["overall_exports"]
    if report is None:
        st.info("完成一次总体测试后，这里会自动出现结论摘要、导出按钮和逐样本结果。")
        return
    frame = pd.DataFrame(report.summary).copy()
    frame["model_label"] = frame["model_id"].map(lambda item: MODEL_LIBRARY.get(item, {}).get("label", item))
    best_uss = frame.sort_values("uss", ascending=False).iloc[0]
    best_cer = frame.sort_values("cer", ascending=True).iloc[0]
    fastest = frame.sort_values("avg_latency_ms", ascending=True).iloc[0]
    cols = st.columns(3)
    cols[0].metric("综合最优", f"{best_uss['model_label']} / {best_uss['uss']:.2f}")
    cols[1].metric("最低 CER", f"{best_cer['model_label']} / {best_cer['cer']:.4f}")
    cols[2].metric("最快延迟", f"{fastest['model_label']} / {fastest['avg_latency_ms']:.2f} ms")
    st.markdown(
        f"""
        <div class="insight">
          <h3>本轮实验结论</h3>
          <p>综合满意度最高的模型是 <strong>{html.escape(str(best_uss["model_label"]))}</strong>，USS 为 <strong>{best_uss["uss"]:.2f}</strong>。</p>
          <p>识别准确性最佳的模型是 <strong>{html.escape(str(best_cer["model_label"]))}</strong>，CER 降至 <strong>{best_cer["cer"]:.4f}</strong>。</p>
          <p>平均延迟最快的模型是 <strong>{html.escape(str(fastest["model_label"]))}</strong>，平均延迟为 <strong>{fastest["avg_latency_ms"]:.2f} ms</strong>。</p>
        </div>
        """,
        unsafe_allow_html=True,
    )
    if exports:
        dcols = st.columns(3)
        dcols[0].download_button("下载 JSON 报告", data=Path(exports["json"]).read_bytes(), file_name=Path(exports["json"]).name, mime="application/json", use_container_width=True)
        dcols[1].download_button("下载 CSV 摘要", data=Path(exports["csv"]).read_bytes(), file_name=Path(exports["csv"]).name, mime="text/csv", use_container_width=True)
        dcols[2].download_button("下载 Markdown 报告", data=Path(exports["markdown"]).read_bytes(), file_name=Path(exports["markdown"]).name, mime="text/markdown", use_container_width=True)
    with st.expander("查看逐样本识别结果", expanded=False):
        st.dataframe(sample_frame(report.sample_results), use_container_width=True, hide_index=True)
    history = list_saved_experiments()
    if history:
        with st.expander("历史实验记录", expanded=False):
            st.dataframe(pd.DataFrame(history), use_container_width=True, hide_index=True)


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


def main() -> None:
    inject_styles()
    history_count = len(list_saved_experiments())
    render_sidebar(history_count)
    render_hero(history_count)
    render_dataset_section()
    st.divider()
    render_model_section()
    st.divider()
    render_evaluation_section()
    st.divider()
    render_results_section()
    st.divider()
    render_chart_section()


if __name__ == "__main__":
    main()
