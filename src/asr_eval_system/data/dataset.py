from __future__ import annotations

import csv
import json
from pathlib import Path

from asr_eval_system.data.audio_utils import read_wave_duration
from asr_eval_system.schemas import DatasetManifest


def load_manifest(path: str | Path) -> list[DatasetManifest]:
    manifest_path = Path(path)
    if manifest_path.suffix.lower() == ".json":
        records = json.loads(manifest_path.read_text(encoding="utf-8"))
    elif manifest_path.suffix.lower() == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as file:
            records = list(csv.DictReader(file))
    else:
        raise ValueError("仅支持 JSON 或 CSV 格式的数据清单。")

    manifests: list[DatasetManifest] = []
    for record in records:
        manifests.append(
            DatasetManifest(
                sample_id=str(record["sample_id"]),
                audio_path=str(record["audio_path"]),
                transcript=str(record["transcript"]),
                duration_sec=float(record.get("duration_sec") or 0),
                split=str(record.get("split") or "test"),
                scene_tag=str(record.get("scene_tag") or "quiet"),
                noise_tag=str(record.get("noise_tag") or "none"),
                accent_tag=str(record.get("accent_tag") or "standard"),
            )
        )
    return manifests


def validate_manifest(items: list[DatasetManifest]) -> list[str]:
    issues: list[str] = []
    sample_ids: set[str] = set()
    for item in items:
        if item.sample_id in sample_ids:
            issues.append(f"存在重复 sample_id: {item.sample_id}")
        sample_ids.add(item.sample_id)

        path = Path(item.audio_path)
        if not path.exists():
            issues.append(f"音频文件不存在: {item.audio_path}")
            continue

        if item.duration_sec <= 0 and path.suffix.lower() == ".wav":
            item.duration_sec = read_wave_duration(path)
        if item.duration_sec <= 0:
            issues.append(f"样本时长无效: {item.sample_id}")
        if not item.transcript.strip():
            issues.append(f"样本文本为空: {item.sample_id}")
    return issues

