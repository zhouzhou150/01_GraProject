from __future__ import annotations

import contextlib
import html
import json
import math
import os
import subprocess
import sys
import time
from collections.abc import Callable
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd
import streamlit as st

from asr_eval_system.data.audio_utils import (
    build_sample_id,
    decode_transcript_bytes,
    normalize_audio_file,
    normalize_uploaded_name,
    resolve_transcript_text,
    transcript_match_keys,
)
from asr_eval_system.data.dataset import load_manifest, validate_manifest
from asr_eval_system.reporting.report_generator import export_report_bundle
from asr_eval_system.runner.evaluation import build_aggregate_report, run_experiment_from_specs
from asr_eval_system.schemas import AggregateReport, DatasetManifest, ExperimentConfig
from asr_eval_system.service import database_path, load_default_profile, manifest_path, report_dir, runtime_dir
from asr_eval_system.storage.database import DatabaseManager
from ui.constants import MODEL_LIBRARY, OPTION_LABELS, SAMPLE_COLUMN_MAP, SUMMARY_COLUMN_MAP


ProgressCallback = Callable[[dict[str, Any]], None]


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


def render_summary_cards(cards: list[dict[str, str]]) -> None:
    cols = st.columns(len(cards), gap="large")
    for col, card in zip(cols, cards, strict=False):
        with col:
            st.markdown(
                f"""
                <div class="summary-card">
                  <div class="summary-card-label">{html.escape(card["label"])}</div>
                  <div class="summary-card-model">{html.escape(card["model"])}</div>
                  <div class="summary-card-value">{html.escape(card["value"])}</div>
                  <div class="summary-card-meta">{html.escape(card["meta"])}</div>
                </div>
                """,
                unsafe_allow_html=True,
            )


def merge_uploaded_entries(*groups: Any) -> list[Any]:
    merged: list[Any] = []
    seen_names: set[str] = set()
    for group in groups:
        if not group:
            continue
        entries = group if isinstance(group, list) else [group]
        for entry in entries:
            normalized_name = normalize_uploaded_name(getattr(entry, "name", ""))
            if normalized_name and normalized_name in seen_names:
                continue
            if normalized_name:
                seen_names.add(normalized_name)
            merged.append(entry)
    return merged


def build_sidecar_transcript_map(uploaded_text_files: list | None) -> dict[str, str]:
    transcript_map: dict[str, str] = {}
    for transcript_file in uploaded_text_files or []:
        content = decode_transcript_bytes(transcript_file.getvalue(), suffix=Path(transcript_file.name).suffix)
        for key in transcript_match_keys(transcript_file.name):
            transcript_map[key] = content
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
        normalized_name = normalize_uploaded_name(uploaded_audio.name)
        source_path = Path(normalized_name)
        target_name = source_path.with_suffix(".wav").name
        audio_path = target_dir / target_name
        if audio_path.exists():
            audio_path = target_dir / f"{audio_path.stem}_{index:02d}.wav"
        status.info(f"正在导入：{source_path.name}")
        raw_path = target_dir / source_path.name
        raw_path.write_bytes(uploaded_audio.getvalue())
        duration_sec = normalize_audio_file(raw_path, audio_path)
        ref_text = resolve_transcript_text(transcript_lookup, normalized_name).strip()
        if ref_text:
            audio_path.with_suffix(".txt").write_text(ref_text, encoding="utf-8")
        items.append(
            DatasetManifest(
                sample_id=build_sample_id(normalized_name, index),
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


def _coerce_str(value: Any) -> str:
    return "" if value is None else str(value)


def _coerce_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "y", "on"}
    return bool(value)


def _coerce_float(value: Any, field_name: str, default: float = 0.0) -> float:
    if value is None:
        return float(default)
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        result = float(value)
    else:
        candidate = value
        if hasattr(candidate, "item"):
            with contextlib.suppress(Exception):
                candidate = candidate.item()
        if hasattr(candidate, "numpy"):
            with contextlib.suppress(Exception):
                array_value = candidate.numpy()
                if getattr(array_value, "size", 0) == 1:
                    candidate = float(array_value.reshape(-1)[0])
        if hasattr(candidate, "tolist"):
            with contextlib.suppress(Exception):
                listed = candidate.tolist()
                if isinstance(listed, list) and len(listed) == 1:
                    candidate = listed[0]
        try:
            result = float(candidate)
        except Exception as exc:
            raise TypeError(f"{field_name} 无法转换为数值，当前类型为 {type(value).__name__}") from exc
    if not math.isfinite(result):
        raise TypeError(f"{field_name} 不是有效数值：{result}")
    return result


def _sanitize_options(options: dict[str, Any] | None) -> dict[str, str]:
    return {_coerce_str(key): _coerce_str(value) for key, value in dict(options or {}).items()}


def _sanitize_model_specs(model_specs: list[dict]) -> list[dict]:
    cleaned: list[dict] = []
    for spec in model_specs:
        model_id = _coerce_str(spec.get("model_id"))
        simulate = _coerce_bool(spec.get("simulate", True))
        if model_id in {"cnn_ctc", "rnn_ctc"}:
            simulate = True
        cleaned.append(
            {
                "model_id": model_id,
                "device": _coerce_str(spec.get("device", "cpu")) or "cpu",
                "simulate": simulate,
                "options": _sanitize_options(spec.get("options")),
            }
        )
    return cleaned


def _sanitize_dataset_items(dataset_items: list[DatasetManifest]) -> list[DatasetManifest]:
    cleaned: list[DatasetManifest] = []
    for item in dataset_items:
        cleaned.append(
            DatasetManifest(
                sample_id=_coerce_str(getattr(item, "sample_id", "")),
                audio_path=_coerce_str(getattr(item, "audio_path", "")),
                transcript=_coerce_str(getattr(item, "transcript", "")),
                duration_sec=_coerce_float(getattr(item, "duration_sec", 0.0), "duration_sec"),
                split=_coerce_str(getattr(item, "split", "test")) or "test",
                scene_tag=_coerce_str(getattr(item, "scene_tag", "quiet")) or "quiet",
                noise_tag=_coerce_str(getattr(item, "noise_tag", "none")) or "none",
                accent_tag=_coerce_str(getattr(item, "accent_tag", "standard")) or "standard",
            )
        )
    return cleaned


def _normalize_progress_state(event: dict[str, Any]) -> dict[str, Any]:
    state = dict(event)
    model_total = max(int(state.get("model_total", 1) or 1), 1)
    model_index = int(state.get("model_index", 1) or 1)
    model_index = max(1, min(model_index, model_total))
    model_total_steps = max(int(state.get("model_total_steps", 1) or 1), 1)
    model_step = int(state.get("model_step", 0) or 0)
    model_step = max(0, min(model_step, model_total_steps))
    sample_total = max(int(state.get("sample_total", 0) or 0), 0)
    sample_index = int(state.get("sample_index", 0) or 0)
    sample_index = max(0, min(sample_index, sample_total if sample_total else sample_index))
    overall_total_steps = model_total * model_total_steps
    overall_completed_steps = min(overall_total_steps, (model_index - 1) * model_total_steps + model_step)

    state.update(
        {
            "stage": _coerce_str(state.get("stage")) or "running",
            "model_id": _coerce_str(state.get("model_id")),
            "model_index": model_index,
            "model_total": model_total,
            "model_step": model_step,
            "model_total_steps": model_total_steps,
            "sample_index": sample_index,
            "sample_total": sample_total,
            "sample_id": _coerce_str(state.get("sample_id")),
            "pred_text": _coerce_str(state.get("pred_text")),
            "ref_text": _coerce_str(state.get("ref_text")),
            "backend": _coerce_str(state.get("backend")),
            "runtime_mode": _coerce_str(state.get("runtime_mode")),
            "status": _coerce_str(state.get("status")) or "running",
            "error_message": _coerce_str(state.get("error_message")),
            "overall_total_steps": overall_total_steps,
            "overall_completed_steps": overall_completed_steps,
        }
    )
    return state


def _run_single_model_in_subprocess(
    config: ExperimentConfig,
    dataset_items: list[DatasetManifest],
    model_spec: dict[str, Any],
    progress_callback: ProgressCallback | None = None,
    model_index: int = 1,
    model_total: int = 1,
) -> AggregateReport:
    job_id = datetime.now().strftime("eval_job_%Y%m%d_%H%M%S_%f")
    job_dir = runtime_dir() / "eval_jobs"
    job_dir.mkdir(parents=True, exist_ok=True)
    payload_path = job_dir / f"{job_id}_payload.json"
    output_path = job_dir / f"{job_id}_output.json"
    status_path = job_dir / f"{job_id}_status.json"
    model_total_steps = max(len(dataset_items) + 2, 2)

    payload = {
        "config": config.to_dict(),
        "dataset_items": [item.to_dict() for item in dataset_items],
        "model_specs": [model_spec],
        "status_path": str(status_path),
    }
    payload_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    command = [
        sys.executable,
        "-m",
        "asr_eval_system.runner.subprocess_worker",
        str(payload_path),
        str(output_path),
    ]
    env = os.environ.copy()
    python_root = Path(sys.executable).resolve().parent
    runtime_paths = [
        str(python_root),
        str(python_root / "Library" / "bin"),
        str(python_root / "Library" / "usr" / "bin"),
        str(python_root / "Scripts"),
    ]
    existing_path_entries = env.get("PATH", "").split(os.pathsep)
    merged_path_entries: list[str] = []
    for entry in runtime_paths + existing_path_entries:
        normalized = entry.strip()
        if normalized and normalized not in merged_path_entries:
            merged_path_entries.append(normalized)
    env["PATH"] = os.pathsep.join(merged_path_entries)

    if progress_callback is not None:
        progress_callback(
            _normalize_progress_state(
                {
                    "stage": "queued",
                    "model_id": model_spec.get("model_id", ""),
                    "model_index": model_index,
                    "model_total": model_total,
                    "model_step": 0,
                    "model_total_steps": model_total_steps,
                    "sample_index": 0,
                    "sample_total": len(dataset_items),
                    "sample_id": "",
                    "pred_text": "",
                    "ref_text": "",
                    "status": "running",
                    "error_message": "",
                }
            )
        )

    process = subprocess.Popen(
        command,
        cwd=str(Path(__file__).resolve().parents[2]),
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        encoding="utf-8",
        errors="replace",
    )

    last_status_payload = ""

    def forward_status(force: bool = False) -> None:
        nonlocal last_status_payload
        if progress_callback is None or not status_path.exists():
            return
        try:
            payload_text = status_path.read_text(encoding="utf-8")
        except OSError:
            return
        if not force and payload_text == last_status_payload:
            return
        last_status_payload = payload_text
        try:
            raw_status = json.loads(payload_text)
        except json.JSONDecodeError:
            return
        raw_status["model_index"] = model_index
        raw_status["model_total"] = model_total
        raw_status.setdefault("model_total_steps", model_total_steps)
        raw_status.setdefault("sample_total", len(dataset_items))
        progress_callback(_normalize_progress_state(raw_status))

    started_at = time.perf_counter()
    while process.poll() is None:
        forward_status()
        if time.perf_counter() - started_at > 3600:
            process.kill()
            raise RuntimeError(f"{model_spec.get('model_id', 'unknown')} 评测超时，已超过 3600 秒。")
        time.sleep(0.2)

    stdout, stderr = process.communicate()
    forward_status(force=True)

    if not output_path.exists():
        detail = stderr.strip() or stdout.strip() or "子进程未返回结果文件。"
        raise RuntimeError(f"真实评测子进程未成功返回结果：{detail}")

    result = json.loads(output_path.read_text(encoding="utf-8"))
    if not result.get("ok"):
        error_type = result.get("error_type", "RuntimeError")
        error_message = result.get("error_message", "未知错误")
        traceback_text = result.get("traceback", "").strip()
        if traceback_text:
            raise RuntimeError(f"{error_type}: {error_message}\n{traceback_text}")
        raise RuntimeError(f"{error_type}: {error_message}")
    return AggregateReport(**result["report"])


def _run_evaluation_in_subprocess(
    config: ExperimentConfig,
    dataset_items: list[DatasetManifest],
    model_specs: list[dict[str, Any]],
    progress_callback: ProgressCallback | None = None,
) -> AggregateReport:
    summary: list[dict[str, Any]] = []
    sample_results: list[dict[str, Any]] = []
    profile = load_default_profile()
    model_total = max(len(model_specs), 1)

    for model_index, spec in enumerate(model_specs, start=1):
        single_config = ExperimentConfig(
            experiment_id=config.experiment_id,
            model_ids=[spec["model_id"]],
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            device=str(spec.get("device", config.device)),
            batch_size=config.batch_size,
            metrics_profile=config.metrics_profile,
            output_dir=config.output_dir,
            notes=config.notes,
        )
        report = _run_single_model_in_subprocess(
            config=single_config,
            dataset_items=dataset_items,
            model_spec=spec,
            progress_callback=progress_callback,
            model_index=model_index,
            model_total=model_total,
        )
        summary.extend(report.summary)
        sample_results.extend(report.sample_results)

    return build_aggregate_report(
        config=config,
        dataset_name=config.dataset_name,
        summary=summary,
        sample_results=sample_results,
        profile=profile,
    )


def run_evaluation_workflow(
    dataset_items: list[DatasetManifest],
    dataset_name: str,
    model_specs: list[dict],
    experiment_prefix: str,
    sample_limit: int | None = None,
    export_bundle: bool = False,
    progress_callback: ProgressCallback | None = None,
) -> tuple[AggregateReport, dict[str, str] | None]:
    clean_items = _sanitize_dataset_items(dataset_items)
    clean_model_specs = _sanitize_model_specs(model_specs)
    clean_sample_limit = int(_coerce_float(sample_limit, "sample_limit", default=0.0)) if sample_limit else None
    selected_items = list(clean_items[:clean_sample_limit] if clean_sample_limit else clean_items)
    config = ExperimentConfig(
        experiment_id=datetime.now().strftime(f"{experiment_prefix}_%Y%m%d_%H%M%S"),
        model_ids=[spec["model_id"] for spec in clean_model_specs],
        dataset_name=_coerce_str(dataset_name) or "temporary_dataset",
        output_dir=str(report_dir()),
        notes=json.dumps(
            [
                {
                    "model_id": spec["model_id"],
                    "device": spec["device"],
                    "simulate": spec["simulate"],
                    "options": spec["options"],
                    "sample_limit": clean_sample_limit,
                }
                for spec in clean_model_specs
            ],
            ensure_ascii=False,
        ),
    )

    if any(not spec["simulate"] for spec in clean_model_specs):
        report = _run_evaluation_in_subprocess(
            config=config,
            dataset_items=selected_items,
            model_specs=clean_model_specs,
            progress_callback=progress_callback,
        )
    else:
        direct_progress = None
        if progress_callback is not None:
            direct_progress = lambda event: progress_callback(_normalize_progress_state(event))
        report = run_experiment_from_specs(
            config=config,
            dataset_items=selected_items,
            model_specs=clean_model_specs,
            profile=load_default_profile(),
            progress_callback=direct_progress,
        )

    exports = None
    if export_bundle:
        exports = export_report_bundle(report, report_dir())
        database = DatabaseManager(database_path())
        database.save_experiment(report)
        for export_type in ("json", "csv", "markdown"):
            database.record_export(report.experiment_id, export_type, exports[export_type], exports["created_at"])
    return report, exports
