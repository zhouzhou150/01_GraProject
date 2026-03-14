from __future__ import annotations

import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class FasterWhisperAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str = "faster_whisper",
        device: str = "cpu",
        simulate: bool = True,
        model_size: str = "base",
    ) -> None:
        super().__init__(model_id=model_id, device=device, simulate=simulate)
        self.model_size = model_size
        self._model = None

    def load(self) -> None:
        start = time.perf_counter()
        if not self.simulate:
            try:
                from faster_whisper import WhisperModel
            except ImportError:
                self.simulate = True
                self.backend_name = "simulated"
            else:  # pragma: no cover
                self._model = WhisperModel(self.model_size, device=self.device)
                self.backend_name = "faster-whisper"
        if self.simulate:
            time.sleep(0.03)
            self.backend_name = "simulated"
        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        if self._model is not None:  # pragma: no cover
            segments, _ = self._model.transcribe(audio_path, language="zh")
            return "".join(segment.text.strip() for segment in segments)
        return build_simulated_prediction(audio_path, error_rate=0.06, seed_bias=37)

