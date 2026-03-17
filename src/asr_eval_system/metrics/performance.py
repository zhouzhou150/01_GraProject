from __future__ import annotations

import contextlib
import math
from statistics import mean

from asr_eval_system.metrics.satisfaction import compute_uss
from asr_eval_system.schemas import AggregateMetrics, InferenceResult, SatisfactionProfile


def aggregate_results(
    model_id: str,
    results: list[InferenceResult],
    profile: SatisfactionProfile,
) -> AggregateMetrics:
    if not results:
        raise ValueError("聚合结果时至少需要一条样本记录。")

    latencies = [_coerce_float(item.latency_ms, "latency_ms") for item in results]
    upls = [_coerce_float(item.upl_ms, "upl_ms") for item in results]
    rtfs = [_coerce_float(item.rtf, "rtf") for item in results]
    throughputs = [_coerce_float(item.throughput, "throughput") for item in results]
    cers = [_coerce_float(item.cer, "cer") for item in results]
    wers = [_coerce_float(item.wer, "wer") for item in results]
    sers = [_coerce_float(item.ser, "ser") for item in results]
    semdists = [_coerce_float(item.semdist, "semdist") for item in results]
    cpus = [_coerce_float(item.cpu_pct, "cpu_pct") for item in results if item.cpu_pct is not None]
    mems = [_coerce_float(item.mem_mb, "mem_mb") for item in results if item.mem_mb is not None]
    gpus = [_coerce_float(item.gpu_mem_mb, "gpu_mem_mb") for item in results if item.gpu_mem_mb is not None]
    load_time_ms = max(_coerce_float(item.load_time_ms, "load_time_ms") for item in results)

    quiet = [item for item in results if item.scene_tag == "quiet"]
    noisy = [item for item in results if item.scene_tag != "quiet"]
    quiet_cer = mean([item.cer for item in quiet]) if quiet else mean(cers)
    noisy_cer = mean([item.cer for item in noisy]) if noisy else quiet_cer
    robustness_ratio = 1.0 if math.isclose(quiet_cer, 0.0) else max(0.0, 1 - (noisy_cer - quiet_cer))
    robustness_score = round(min(100.0, max(0.0, robustness_ratio * 100)), 4)

    avg_cer = round(mean(cers), 4)
    avg_wer = round(mean(wers), 4)
    avg_ser = round(mean(sers), 4)
    avg_semdist = round(mean(semdists), 4)
    avg_latency = round(mean(latencies), 4)
    avg_upl = round(mean(upls), 4)
    avg_rtf = round(mean(rtfs), 4)
    throughput = round(mean(throughputs), 4)
    p95_latency = round(_percentile(latencies, 95), 4)
    cpu_value = round(mean(cpus), 4) if cpus else None
    mem_value = round(mean(mems), 4) if mems else None
    gpu_value = round(mean(gpus), 4) if gpus else None

    uss, _, level = compute_uss(
        cer_value=avg_cer,
        semdist_value=avg_semdist,
        upl_ms=avg_upl,
        rtf_value=avg_rtf,
        robustness_score=robustness_score,
        cpu_pct=cpu_value,
        mem_mb=mem_value,
        gpu_mem_mb=gpu_value,
        load_time_ms=load_time_ms,
        profile=profile,
    )

    return AggregateMetrics(
        model_id=model_id,
        backend=results[0].backend,
        runtime_mode=results[0].runtime_mode,
        sample_count=len(results),
        cer=avg_cer,
        wer=avg_wer,
        ser=avg_ser,
        semdist=avg_semdist,
        avg_latency_ms=avg_latency,
        p95_latency_ms=p95_latency,
        avg_upl_ms=avg_upl,
        avg_rtf=avg_rtf,
        throughput=throughput,
        cpu_pct=cpu_value,
        mem_mb=mem_value,
        gpu_mem_mb=gpu_value,
        load_time_ms=load_time_ms,
        robustness_score=robustness_score,
        resource_score=_resource_proxy(cpu_value, mem_value, gpu_value, load_time_ms),
        uss=uss,
        satisfaction_level=level,
    )


def _percentile(values: list[float], percentile: int) -> float:
    ordered = sorted(values)
    index = max(0, min(len(ordered) - 1, math.ceil((percentile / 100) * len(ordered)) - 1))
    return ordered[index]


def _coerce_float(value, field_name: str) -> float:
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
        raise TypeError(f"{field_name} 必须是数值类型，当前为 {type(value).__name__}") from exc


def _resource_proxy(
    cpu_pct: float | None,
    mem_mb: float | None,
    gpu_mem_mb: float | None,
    load_time_ms: float,
) -> float:
    cpu_score = 100 - min(cpu_pct or 0.0, 100.0)
    mem_score = 100 - min((mem_mb or 0.0) / 40.96, 100.0)
    gpu_score = 100 - min((gpu_mem_mb or 0.0) / 81.92, 100.0)
    load_score = 100 - min(load_time_ms / 50.0, 100.0)
    return round((cpu_score + mem_score + gpu_score + load_score) / 4, 4)
