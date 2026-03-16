from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime
from typing import Any


@dataclass(slots=True)
class DatasetManifest:
    sample_id: str
    audio_path: str
    transcript: str
    duration_sec: float
    split: str = "test"
    scene_tag: str = "quiet"
    noise_tag: str = "none"
    accent_tag: str = "standard"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str
    model_ids: list[str]
    dataset_name: str
    dataset_split: str = "test"
    device: str = "cpu"
    batch_size: int = 1
    metrics_profile: str = "default"
    output_dir: str = "data/reports"
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class InferenceResult:
    sample_id: str
    model_id: str
    backend: str
    runtime_mode: str
    pred_text: str
    ref_text: str
    latency_ms: float
    upl_ms: float
    rtf: float
    throughput: float
    cpu_pct: float | None
    mem_mb: float | None
    gpu_mem_mb: float | None
    load_time_ms: float
    cer: float
    wer: float
    ser: float
    semdist: float
    scene_tag: str
    noise_tag: str
    accent_tag: str
    status: str = "ok"
    error_message: str = ""

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AggregateMetrics:
    model_id: str
    backend: str
    runtime_mode: str
    sample_count: int
    cer: float
    wer: float
    ser: float
    semdist: float
    avg_latency_ms: float
    p95_latency_ms: float
    avg_upl_ms: float
    avg_rtf: float
    throughput: float
    cpu_pct: float | None
    mem_mb: float | None
    gpu_mem_mb: float | None
    load_time_ms: float
    robustness_score: float
    resource_score: float
    uss: float
    satisfaction_level: str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class SatisfactionProfile:
    lit_weights: dict[str, float]
    survey_weights: dict[str, float] | None
    final_weights: dict[str, float]
    good_bad_thresholds: dict[str, float]
    score_mode: str
    source_notes: list[str] = field(default_factory=list)
    survey_blend_ratio: float = 0.30

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class AggregateReport:
    experiment_id: str
    dataset_name: str
    created_at: str
    config: dict[str, Any]
    summary: list[dict[str, Any]]
    sample_results: list[dict[str, Any]]
    satisfaction_profile: dict[str, Any]
    charts: dict[str, Any] = field(default_factory=dict)
    conclusion_text: str = ""

    @classmethod
    def build(
        cls,
        experiment_id: str,
        dataset_name: str,
        config: dict[str, Any],
        summary: list[dict[str, Any]],
        sample_results: list[dict[str, Any]],
        satisfaction_profile: dict[str, Any],
        charts: dict[str, Any] | None = None,
        conclusion_text: str = "",
    ) -> "AggregateReport":
        return cls(
            experiment_id=experiment_id,
            dataset_name=dataset_name,
            created_at=datetime.now().isoformat(timespec="seconds"),
            config=config,
            summary=summary,
            sample_results=sample_results,
            satisfaction_profile=satisfaction_profile,
            charts=charts or {},
            conclusion_text=conclusion_text,
        )

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
