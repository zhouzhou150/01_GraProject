from __future__ import annotations

from datetime import datetime
from pathlib import Path

from asr_eval_system.config.settings import load_satisfaction_profile
from asr_eval_system.data.dataset import load_manifest, validate_manifest
from asr_eval_system.models.registry import build_model_registry
from asr_eval_system.reporting.report_generator import export_report_bundle
from asr_eval_system.runner.evaluation import run_experiment
from asr_eval_system.schemas import AggregateReport, ExperimentConfig
from asr_eval_system.storage.database import DatabaseManager


def project_root() -> Path:
    return Path(__file__).resolve().parents[2]


def runtime_dir() -> Path:
    path = project_root() / "data" / "runtime"
    path.mkdir(parents=True, exist_ok=True)
    return path


def report_dir() -> Path:
    path = project_root() / "data" / "reports"
    path.mkdir(parents=True, exist_ok=True)
    return path


def manifest_path() -> Path:
    return project_root() / "data" / "manifests" / "demo_manifest.json"


def database_path() -> Path:
    return runtime_dir() / "asr_eval.db"


def load_default_profile():
    return load_satisfaction_profile(project_root() / "config" / "satisfaction_profile.yml")


def validate_default_manifest() -> tuple[list, list[str]]:
    items = load_manifest(manifest_path())
    issues = validate_manifest(items)
    return items, issues


def run_default_experiment(
    model_ids: list[str] | None = None,
    simulate: bool = True,
    experiment_id: str | None = None,
) -> tuple[AggregateReport, dict[str, str]]:
    items, issues = validate_default_manifest()
    if issues:
        raise ValueError("默认数据清单校验失败: " + "; ".join(issues))

    config = ExperimentConfig(
        experiment_id=experiment_id or datetime.now().strftime("exp_%Y%m%d_%H%M%S"),
        model_ids=model_ids or ["cnn_ctc", "rnn_ctc", "faster_whisper", "paddlespeech"],
        dataset_name="demo_manifest",
        output_dir=str(report_dir()),
    )
    profile = load_default_profile()
    report = run_experiment(config, items, build_model_registry(simulate=simulate), profile)
    exports = export_report_bundle(report, report_dir())
    database = DatabaseManager(database_path())
    database.save_experiment(report)
    database.record_export(report.experiment_id, "bundle", exports["markdown"], exports["created_at"])
    return report, exports


def list_saved_experiments() -> list[dict[str, str]]:
    return DatabaseManager(database_path()).list_experiments()


def load_saved_report(experiment_id: str):
    return DatabaseManager(database_path()).get_report(experiment_id)

