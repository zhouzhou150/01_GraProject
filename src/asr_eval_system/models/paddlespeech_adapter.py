from __future__ import annotations

import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class PaddleSpeechAdapter(ModelAdapter):
    def __init__(self, model_id: str = "paddlespeech", device: str = "cpu", simulate: bool = True) -> None:
        super().__init__(model_id=model_id, device=device, simulate=simulate)
        self._executor = None

    def load(self) -> None:
        start = time.perf_counter()
        if not self.simulate:
            try:
                from paddlespeech.cli.asr.infer import ASRExecutor
            except ImportError:
                self.simulate = True
                self.backend_name = "simulated"
            else:  # pragma: no cover
                self._executor = ASRExecutor()
                self.backend_name = "paddlespeech"
        if self.simulate:
            time.sleep(0.04)
            self.backend_name = "simulated"
        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        if self._executor is not None:  # pragma: no cover
            return str(self._executor(audio_file=audio_path, lang="zh"))
        return build_simulated_prediction(audio_path, error_rate=0.09, seed_bias=41)

