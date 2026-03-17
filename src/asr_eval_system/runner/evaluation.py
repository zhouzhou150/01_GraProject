from __future__ import annotations

import contextlib
import time
from collections.abc import Callable
from typing import Any

from asr_eval_system.metrics.performance import aggregate_results
from asr_eval_system.metrics.text_metrics import cer, semdist_score, ser, wer
from asr_eval_system.models.registry import build_model_registry_from_specs
from asr_eval_system.schemas import AggregateReport, DatasetManifest, ExperimentConfig, InferenceResult, SatisfactionProfile


ProgressCallback = Callable[[dict[str, Any]], None]

_PROCESS_LIFETIME_ADAPTERS: list[object] = []


def _coerce_float(value, field_name: str, allow_none: bool = False) -> float | None:
    if value is None:
        return None if allow_none else 0.0
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
        return float(candidate)
    except Exception as exc:
        raise TypeError(f"{field_name} must be a real number, got {type(value).__name__}") from exc


def _emit_progress(progress_callback: ProgressCallback | None, **payload: Any) -> None:
    if progress_callback is not None:
        progress_callback(payload)


def run_experiment(
    config: ExperimentConfig,
    dataset_items: list[DatasetManifest],
    model_registry: dict[str, object],
    profile: SatisfactionProfile,
    skip_unload_model_ids: set[str] | None = None,
    progress_callback: ProgressCallback | None = None,
) -> AggregateReport:
    sample_results: list[dict] = []
    summary: list[dict] = []
    skip_unload = {str(model_id) for model_id in (skip_unload_model_ids or set())}
    model_total = max(len(config.model_ids), 1)
    sample_total = len(dataset_items)
    model_total_steps = max(sample_total + 2, 2)

    for model_index, model_id in enumerate(config.model_ids, start=1):
        adapter = model_registry[model_id]

        _emit_progress(
            progress_callback,
            stage="loading",
            model_id=model_id,
            model_index=model_index,
            model_total=model_total,
            model_step=0,
            model_total_steps=model_total_steps,
            sample_index=0,
            sample_total=sample_total,
            sample_id="",
            pred_text="",
            ref_text="",
            status="running",
            error_message="",
        )

        try:
            adapter.load()
            metadata = adapter.metadata()
            _emit_progress(
                progress_callback,
                stage="warmup",
                model_id=model_id,
                model_index=model_index,
                model_total=model_total,
                model_step=1,
                model_total_steps=model_total_steps,
                sample_index=0,
                sample_total=sample_total,
                sample_id="",
                pred_text="",
                ref_text="",
                backend=str(metadata.get("backend", "unknown")),
                runtime_mode="模拟" if bool(metadata.get("simulate", True)) else "真实",
                status="running",
                error_message="",
            )

            warmup_audio = dataset_items[0].audio_path if dataset_items else None
            adapter.warmup(warmup_audio)
            metadata = adapter.metadata()
            _emit_progress(
                progress_callback,
                stage="running",
                model_id=model_id,
                model_index=model_index,
                model_total=model_total,
                model_step=2,
                model_total_steps=model_total_steps,
                sample_index=0,
                sample_total=sample_total,
                sample_id="",
                pred_text="",
                ref_text="",
                backend=str(metadata.get("backend", "unknown")),
                runtime_mode="模拟" if bool(metadata.get("simulate", True)) else "真实",
                status="running",
                error_message="",
            )
        except Exception as exc:
            metadata = adapter.metadata()
            _emit_progress(
                progress_callback,
                stage="error",
                model_id=model_id,
                model_index=model_index,
                model_total=model_total,
                model_step=0,
                model_total_steps=model_total_steps,
                sample_index=0,
                sample_total=sample_total,
                sample_id="",
                pred_text="",
                ref_text="",
                backend=str(metadata.get("backend", "unknown")),
                runtime_mode="模拟" if bool(metadata.get("simulate", True)) else "真实",
                status="error",
                error_message=str(exc),
            )
            raise

        model_results: list[InferenceResult] = []

        for item_index, item in enumerate(dataset_items, start=1):
            start = time.perf_counter()
            cpu_pct, mem_mb, gpu_mem_mb = _resource_snapshot()
            try:
                prediction = adapter.transcribe(item.audio_path)
                infer_elapsed_ms = (time.perf_counter() - start) * 1000
                upl_ms = infer_elapsed_ms + 30.0
                duration = max(_coerce_float(item.duration_sec, f"{item.sample_id}.duration_sec") or 0.0, 0.1)
                rtf_value = (infer_elapsed_ms / 1000) / duration
                throughput = 1000 / max(infer_elapsed_ms, 1.0)
                cpu_value = _coerce_float(cpu_pct, f"{item.sample_id}.cpu_pct", allow_none=True)
                mem_value = _coerce_float(mem_mb, f"{item.sample_id}.mem_mb", allow_none=True)
                gpu_value = _coerce_float(gpu_mem_mb, f"{item.sample_id}.gpu_mem_mb", allow_none=True)
                load_time_ms = _coerce_float(metadata.get("load_time_ms", 0.0), f"{model_id}.load_time_ms") or 0.0
                result = InferenceResult(
                    sample_id=item.sample_id,
                    model_id=model_id,
                    backend=str(metadata.get("backend", "unknown")),
                    runtime_mode="模拟" if bool(metadata.get("simulate", True)) else "真实",
                    pred_text=prediction,
                    ref_text=item.transcript,
                    latency_ms=round(infer_elapsed_ms, 4),
                    upl_ms=round(upl_ms, 4),
                    rtf=round(rtf_value, 4),
                    throughput=round(throughput, 4),
                    cpu_pct=cpu_value,
                    mem_mb=mem_value,
                    gpu_mem_mb=gpu_value,
                    load_time_ms=load_time_ms,
                    cer=round(cer(item.transcript, prediction), 4),
                    wer=round(wer(item.transcript, prediction), 4),
                    ser=round(ser(item.transcript, prediction), 4),
                    semdist=round(semdist_score(item.transcript, prediction), 4),
                    scene_tag=item.scene_tag,
                    noise_tag=item.noise_tag,
                    accent_tag=item.accent_tag,
                )
            except Exception as exc:  # pragma: no cover
                infer_elapsed_ms = (time.perf_counter() - start) * 1000
                cpu_value = _coerce_float(cpu_pct, f"{item.sample_id}.cpu_pct", allow_none=True)
                mem_value = _coerce_float(mem_mb, f"{item.sample_id}.mem_mb", allow_none=True)
                gpu_value = _coerce_float(gpu_mem_mb, f"{item.sample_id}.gpu_mem_mb", allow_none=True)
                load_time_ms = _coerce_float(metadata.get("load_time_ms", 0.0), f"{model_id}.load_time_ms") or 0.0
                result = InferenceResult(
                    sample_id=item.sample_id,
                    model_id=model_id,
                    backend=str(metadata.get("backend", "unknown")),
                    runtime_mode="模拟" if bool(metadata.get("simulate", True)) else "真实",
                    pred_text="",
                    ref_text=item.transcript,
                    latency_ms=round(infer_elapsed_ms, 4),
                    upl_ms=round(infer_elapsed_ms, 4),
                    rtf=0.0,
                    throughput=0.0,
                    cpu_pct=cpu_value,
                    mem_mb=mem_value,
                    gpu_mem_mb=gpu_value,
                    load_time_ms=load_time_ms,
                    cer=1.0,
                    wer=1.0,
                    ser=1.0,
                    semdist=0.0,
                    scene_tag=item.scene_tag,
                    noise_tag=item.noise_tag,
                    accent_tag=item.accent_tag,
                    status="error",
                    error_message=str(exc),
                )
            model_results.append(result)
            sample_results.append(result.to_dict())
            _emit_progress(
                progress_callback,
                stage="running",
                model_id=model_id,
                model_index=model_index,
                model_total=model_total,
                model_step=min(model_total_steps, 2 + item_index),
                model_total_steps=model_total_steps,
                sample_index=item_index,
                sample_total=sample_total,
                sample_id=item.sample_id,
                pred_text=result.pred_text,
                ref_text=item.transcript,
                backend=result.backend,
                runtime_mode=result.runtime_mode,
                status=result.status,
                error_message=result.error_message,
            )

        if model_id not in skip_unload:
            adapter.unload()
        summary_item = aggregate_results(model_id=model_id, results=model_results, profile=profile)
        summary.append(summary_item.to_dict())
        _emit_progress(
            progress_callback,
            stage="finished",
            model_id=model_id,
            model_index=model_index,
            model_total=model_total,
            model_step=model_total_steps,
            model_total_steps=model_total_steps,
            sample_index=sample_total,
            sample_total=sample_total,
            sample_id=model_results[-1].sample_id if model_results else "",
            pred_text=model_results[-1].pred_text if model_results else "",
            ref_text=model_results[-1].ref_text if model_results else "",
            backend=summary_item.backend,
            runtime_mode=summary_item.runtime_mode,
            status="ok",
            error_message="",
        )

    _retain_adapters(model_registry, skip_unload)

    return AggregateReport.build(
        experiment_id=config.experiment_id,
        dataset_name=config.dataset_name,
        config=config.to_dict(),
        summary=summary,
        sample_results=sample_results,
        satisfaction_profile=profile.to_dict(),
        charts=_build_chart_payload(summary),
        conclusion_text=_build_conclusion(summary),
    )


def run_experiment_from_specs(
    config: ExperimentConfig,
    dataset_items: list[DatasetManifest],
    model_specs: list[dict],
    profile: SatisfactionProfile,
    progress_callback: ProgressCallback | None = None,
) -> AggregateReport:
    summary: list[dict] = []
    sample_results: list[dict] = []
    model_total = max(len(model_specs), 1)

    for model_index, spec in enumerate(model_specs, start=1):
        model_id = str(spec.get("model_id", "unknown"))
        single_config = ExperimentConfig(
            experiment_id=config.experiment_id,
            model_ids=[model_id],
            dataset_name=config.dataset_name,
            dataset_split=config.dataset_split,
            device=str(spec.get("device", config.device)),
            batch_size=config.batch_size,
            metrics_profile=config.metrics_profile,
            output_dir=config.output_dir,
            notes=config.notes,
        )
        skip_unload = {model_id} if _should_isolate_model_spec(spec) else None

        def emit_single_model_progress(event: dict[str, Any]) -> None:
            if progress_callback is None:
                return
            enriched = dict(event)
            enriched["model_index"] = model_index
            enriched["model_total"] = model_total
            progress_callback(enriched)

        try:
            report = run_experiment(
                config=single_config,
                dataset_items=dataset_items,
                model_registry=build_model_registry_from_specs([spec]),
                profile=profile,
                skip_unload_model_ids=skip_unload,
                progress_callback=emit_single_model_progress,
            )
        except Exception as exc:
            raise RuntimeError(f"{model_id} 评测失败：{exc}") from exc
        summary.extend(report.summary)
        sample_results.extend(report.sample_results)

    return build_aggregate_report(
        config=config,
        dataset_name=config.dataset_name,
        summary=summary,
        sample_results=sample_results,
        profile=profile,
    )


def build_aggregate_report(
    config: ExperimentConfig,
    dataset_name: str,
    summary: list[dict],
    sample_results: list[dict],
    profile: SatisfactionProfile,
) -> AggregateReport:
    return AggregateReport.build(
        experiment_id=config.experiment_id,
        dataset_name=dataset_name,
        config=config.to_dict(),
        summary=summary,
        sample_results=sample_results,
        satisfaction_profile=profile.to_dict(),
        charts=_build_chart_payload(summary),
        conclusion_text=_build_conclusion(summary),
    )


def _resource_snapshot() -> tuple[float | None, float | None, float | None]:
    try:
        import psutil
    except ImportError:
        return None, None, None
    process = psutil.Process()
    mem_mb = round(process.memory_info().rss / 1024 / 1024, 4)
    cpu_pct = round(psutil.cpu_percent(interval=None), 4)
    return cpu_pct, mem_mb, None


def _should_isolate_model_spec(model_spec: dict) -> bool:
    return (
        str(model_spec.get("model_id", "")) == "faster_whisper"
        and str(model_spec.get("device", "cpu")) == "cuda"
        and not bool(model_spec.get("simulate", True))
    )


def _retain_adapters(model_registry: dict[str, object], retained_model_ids: set[str]) -> None:
    for model_id in retained_model_ids:
        adapter = model_registry.get(model_id)
        if adapter is not None:
            _PROCESS_LIFETIME_ADAPTERS.append(adapter)


def _build_chart_payload(summary: list[dict]) -> dict[str, list[dict]]:
    return {
        "uss_ranking": sorted(
            [{"model_id": item["model_id"], "uss": item["uss"]} for item in summary],
            key=lambda item: item["uss"],
            reverse=True,
        ),
        "latency_vs_cer": [
            {"model_id": item["model_id"], "avg_latency_ms": item["avg_latency_ms"], "cer": item["cer"]}
            for item in summary
        ],
    }


def _build_conclusion(summary: list[dict]) -> str:
    if not summary:
        return "暂无实验结果。"
    ranking_source = [item for item in summary if item.get("runtime_mode") == "真实"] or summary
    best_uss = max(ranking_source, key=lambda item: item["uss"])
    best_cer = min(ranking_source, key=lambda item: item["cer"])
    fastest = min(ranking_source, key=lambda item: item["avg_latency_ms"])
    prefix = "在真实模型中，" if ranking_source is not summary else ""
    return (
        f"{prefix}综合满意度最高的模型是 {best_uss['model_id']}，USS 为 {best_uss['uss']:.2f}。"
        f"识别误差最低的模型是 {best_cer['model_id']}，CER 为 {best_cer['cer']:.4f}。"
        f"平均延迟最低的模型是 {fastest['model_id']}，平均延迟为 {fastest['avg_latency_ms']:.2f} ms。"
    )
