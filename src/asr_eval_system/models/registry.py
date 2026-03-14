from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

from asr_eval_system.models.cnn_ctc import CNNCTCAdapter
from asr_eval_system.models.faster_whisper_adapter import FasterWhisperAdapter
from asr_eval_system.models.paddlespeech_adapter import PaddleSpeechAdapter
from asr_eval_system.models.rnn_ctc import RNNCTCAdapter


DEFAULT_MODEL_IDS = ["cnn_ctc", "rnn_ctc", "faster_whisper", "paddlespeech"]


def build_model_adapter(
    model_id: str,
    device: str = "cpu",
    simulate: bool = True,
    options: Mapping[str, Any] | None = None,
) -> object:
    config = dict(options or {})
    if model_id == "cnn_ctc":
        return CNNCTCAdapter("cnn_ctc", device=device, simulate=simulate)
    if model_id == "rnn_ctc":
        return RNNCTCAdapter("rnn_ctc", device=device, simulate=simulate)
    if model_id == "faster_whisper":
        return FasterWhisperAdapter(
            device=device,
            simulate=simulate,
            model_size=str(config.get("model_size", "base")),
        )
    if model_id == "paddlespeech":
        return PaddleSpeechAdapter(device=device, simulate=simulate)
    raise KeyError(f"Unknown model_id: {model_id}")


def build_model_registry(
    device: str = "cpu",
    simulate: bool = True,
    model_ids: Sequence[str] | None = None,
    options_by_model: Mapping[str, Mapping[str, Any]] | None = None,
) -> dict[str, object]:
    selected_model_ids = list(model_ids or DEFAULT_MODEL_IDS)
    option_map = dict(options_by_model or {})
    return {
        model_id: build_model_adapter(
            model_id,
            device=device,
            simulate=simulate,
            options=option_map.get(model_id),
        )
        for model_id in selected_model_ids
    }


def build_model_registry_from_specs(model_specs: Sequence[Mapping[str, Any]]) -> dict[str, object]:
    return {
        str(spec["model_id"]): build_model_adapter(
            str(spec["model_id"]),
            device=str(spec.get("device", "cpu")),
            simulate=bool(spec.get("simulate", True)),
            options=spec.get("options"),
        )
        for spec in model_specs
    }
