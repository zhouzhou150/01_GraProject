from __future__ import annotations

import json
import sys
import time
import traceback
from pathlib import Path
from typing import Any

from asr_eval_system.runner.evaluation import run_experiment_from_specs
from asr_eval_system.schemas import DatasetManifest, ExperimentConfig
from asr_eval_system.service import load_default_profile


def _write_json_atomic(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(f"{path.suffix}.tmp")
    temp_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    temp_path.replace(path)


def main(argv: list[str] | None = None) -> int:
    args = list(sys.argv[1:] if argv is None else argv)
    if len(args) != 2:
        print("Usage: python -m asr_eval_system.runner.subprocess_worker <payload.json> <output.json>", file=sys.stderr)
        return 2

    payload_path = Path(args[0])
    output_path = Path(args[1])
    status_path: Path | None = None

    def emit_progress(event: dict[str, Any]) -> None:
        if status_path is None:
            return
        payload = dict(event)
        payload["updated_at"] = time.time()
        _write_json_atomic(status_path, payload)

    try:
        payload = json.loads(payload_path.read_text(encoding="utf-8"))
        raw_status_path = payload.get("status_path")
        status_path = Path(raw_status_path) if raw_status_path else None
        config = ExperimentConfig(**payload["config"])
        dataset_items = [DatasetManifest(**item) for item in payload["dataset_items"]]
        model_specs = list(payload["model_specs"])
        report = run_experiment_from_specs(
            config=config,
            dataset_items=dataset_items,
            model_specs=model_specs,
            profile=load_default_profile(),
            progress_callback=emit_progress,
        )
        result = {"ok": True, "report": report.to_dict()}
    except Exception as exc:  # pragma: no cover
        if status_path is not None:
            emit_progress(
                {
                    "stage": "error",
                    "status": "error",
                    "error_type": type(exc).__name__,
                    "error_message": str(exc),
                }
            )
        result = {
            "ok": False,
            "error_type": type(exc).__name__,
            "error_message": str(exc),
            "traceback": traceback.format_exc(),
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return 0 if result.get("ok") else 1


if __name__ == "__main__":
    raise SystemExit(main())
