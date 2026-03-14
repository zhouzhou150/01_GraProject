from __future__ import annotations

import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class CNNCTCAdapter(ModelAdapter):
    def load(self) -> None:
        start = time.perf_counter()
        time.sleep(0.02)
        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.backend_name = "simulated" if self.simulate else "cnn_ctc"
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        return build_simulated_prediction(audio_path, error_rate=0.16, seed_bias=11)

