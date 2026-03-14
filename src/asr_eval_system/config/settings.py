from __future__ import annotations

from pathlib import Path

from asr_eval_system.metrics.satisfaction import build_satisfaction_profile
from asr_eval_system.schemas import SatisfactionProfile


def load_satisfaction_profile(path: str | Path) -> SatisfactionProfile:
    try:
        import yaml
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("缺少 pyyaml，请先安装项目依赖。") from exc

    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as file:
        raw = yaml.safe_load(file) or {}
    return build_satisfaction_profile(raw)

