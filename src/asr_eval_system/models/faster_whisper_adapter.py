from __future__ import annotations

import importlib
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
        compute_type: str = "int8",
        language: str = "zh",
    ) -> None:
        super().__init__(model_id=model_id, device=device, simulate=simulate)
        self.model_size = model_size
        self.compute_type = compute_type
        self.language = language
        self._model = None

    def load(self) -> None:
        start = time.perf_counter()
        self.load_error = ""
        self._model = None

        try:
            if self.requested_simulate:
                self.simulate = True
                self.backend_name = "simulated"
                self.backend_detail = f"demo backend / {self.model_size}"
                time.sleep(0.03)
            else:
                try:
                    whisper_module = importlib.import_module("faster_whisper")
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装 faster-whisper。请在 Python 3.11 环境中安装 `faster-whisper` 后再使用真实模式。"
                    ) from exc

                whisper_model_cls = getattr(whisper_module, "WhisperModel")
                self._model = whisper_model_cls(
                    self.model_size,
                    device=self.device,
                    compute_type=self.compute_type,
                )
                self.simulate = False
                self.backend_name = "faster-whisper"
                self.backend_detail = f"{self.model_size} / {self.compute_type}"
        except Exception as exc:
            self.load_error = str(exc)
            self.loaded = False
            self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
            raise

        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        if self._model is not None:  # pragma: no cover
            segments, _ = self._model.transcribe(audio_path, language=self.language, task="transcribe")
            return "".join(segment.text.strip() for segment in segments)
        return build_simulated_prediction(audio_path, error_rate=0.06, seed_bias=37)

    def unload(self) -> None:
        self._model = None
        super().unload()
