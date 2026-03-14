from __future__ import annotations

import time

from asr_eval_system.metrics.performance import aggregate_results
from asr_eval_system.metrics.text_metrics import cer, semdist_score, ser, wer
from asr_eval_system.schemas import AggregateReport, DatasetManifest, ExperimentConfig, InferenceResult, SatisfactionProfile


def run_experiment(
    config: ExperimentConfig,
    dataset_items: list[DatasetManifest],
    model_registry: dict[str, object],
    profile: SatisfactionProfile,
) -> AggregateReport:
    sample_results: list[dict] = []
    summary: list[dict] = []

    for model_id in config.model_ids:
        adapter = model_registry[model_id]
        adapter.load()
        adapter.warmup()
        model_results: list[InferenceResult] = []

        for item in dataset_items:
            start = time.perf_counter()
            cpu_pct, mem_mb, gpu_mem_mb = _resource_snapshot()
            try:
                prediction = adapter.transcribe(item.audio_path)
                infer_elapsed_ms = (time.perf_counter() - start) * 1000
                upl_ms = infer_elapsed_ms + 30.0
                duration = max(item.duration_sec, 0.1)
                rtf_value = (infer_elapsed_ms / 1000) / duration
                throughput = 1000 / max(infer_elapsed_ms, 1.0)
                result = InferenceResult(
                    sample_id=item.sample_id,
                    model_id=model_id,
                    pred_text=prediction,
                    ref_text=item.transcript,
                    latency_ms=round(infer_elapsed_ms, 4),
                    upl_ms=round(upl_ms, 4),
                    rtf=round(rtf_value, 4),
                    throughput=round(throughput, 4),
                    cpu_pct=cpu_pct,
                    mem_mb=mem_mb,
                    gpu_mem_mb=gpu_mem_mb,
                    load_time_ms=adapter.metadata().get("load_time_ms", 0.0),
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
                result = InferenceResult(
                    sample_id=item.sample_id,
                    model_id=model_id,
                    pred_text="",
                    ref_text=item.transcript,
                    latency_ms=round(infer_elapsed_ms, 4),
                    upl_ms=round(infer_elapsed_ms, 4),
                    rtf=0.0,
                    throughput=0.0,
                    cpu_pct=cpu_pct,
                    mem_mb=mem_mb,
                    gpu_mem_mb=gpu_mem_mb,
                    load_time_ms=adapter.metadata().get("load_time_ms", 0.0),
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

        adapter.unload()
        summary_item = aggregate_results(model_id=model_id, results=model_results, profile=profile)
        summary.append(summary_item.to_dict())

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


def _resource_snapshot() -> tuple[float | None, float | None, float | None]:
    try:
        import psutil
    except ImportError:
        return None, None, None
    process = psutil.Process()
    mem_mb = round(process.memory_info().rss / 1024 / 1024, 4)
    cpu_pct = round(psutil.cpu_percent(interval=None), 4)
    return cpu_pct, mem_mb, None


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
    best_uss = max(summary, key=lambda item: item["uss"])
    best_cer = min(summary, key=lambda item: item["cer"])
    fastest = min(summary, key=lambda item: item["avg_latency_ms"])
    return (
        f"综合满意度最高的模型为 {best_uss['model_id']}，USS 为 {best_uss['uss']:.2f}。"
        f"识别误差最低的模型为 {best_cer['model_id']}，CER 为 {best_cer['cer']:.4f}。"
        f"平均延迟最低的模型为 {fastest['model_id']}，平均延迟为 {fastest['avg_latency_ms']:.2f} ms。"
    )

