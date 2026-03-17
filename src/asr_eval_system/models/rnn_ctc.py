from __future__ import annotations

import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class RNNCTCAdapter(ModelAdapter):
    def load(self) -> None:
        if self.loaded:
            return
        start = time.perf_counter()
        time.sleep(0.025)
        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.backend_name = "simulated" if self.simulate else "rnn_ctc"
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        return build_simulated_prediction(audio_path, error_rate=0.12, seed_bias=23)
