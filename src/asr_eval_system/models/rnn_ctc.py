from __future__ import annotations

import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class RNNCTCAdapter(ModelAdapter):
    def __init__(self, model_id: str = "rnn_ctc", device: str = "cpu", simulate: bool = True) -> None:
        super().__init__(model_id=model_id, device=device, simulate=simulate)

    def load(self) -> None:
        if self.loaded:
            return
        start = time.perf_counter()
        time.sleep(0.025)
        self.simulate = True
        self.backend_name = "proxy-baseline"
        self.backend_detail = "rnn_ctc proxy / simulated"
        if self.requested_simulate:
            self.runtime_note = "当前使用 RNN-CTC 代理基线，用于轻量对照。"
        else:
            self.runtime_note = "当前项目尚未接入真实 RNN-CTC 模型，已自动回退为代理基线进行对照。"
        self.load_time_ms = round((time.perf_counter() - start) * 1000, 3)
        self.loaded = True

    def transcribe(self, audio_path: str) -> str:
        if not self.loaded:
            self.load()
        return build_simulated_prediction(audio_path, error_rate=0.12, seed_bias=23)
