from __future__ import annotations

from asr_eval_system.models.cnn_ctc import CNNCTCAdapter
from asr_eval_system.models.faster_whisper_adapter import FasterWhisperAdapter
from asr_eval_system.models.paddlespeech_adapter import PaddleSpeechAdapter
from asr_eval_system.models.rnn_ctc import RNNCTCAdapter


def build_model_registry(device: str = "cpu", simulate: bool = True) -> dict[str, object]:
    return {
        "cnn_ctc": CNNCTCAdapter("cnn_ctc", device=device, simulate=simulate),
        "rnn_ctc": RNNCTCAdapter("rnn_ctc", device=device, simulate=simulate),
        "faster_whisper": FasterWhisperAdapter(device=device, simulate=simulate),
        "paddlespeech": PaddleSpeechAdapter(device=device, simulate=simulate),
    }

