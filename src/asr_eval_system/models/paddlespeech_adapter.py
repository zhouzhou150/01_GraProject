from __future__ import annotations

import contextlib
import importlib
import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


class PaddleSpeechAdapter(ModelAdapter):
    def __init__(
        self,
        model_id: str = "paddlespeech",
        device: str = "cpu",
        simulate: bool = True,
        lang: str = "zh",
        postprocess: str = "punctuation",
    ) -> None:
        super().__init__(model_id=model_id, device=device, simulate=simulate)
        self.lang = lang
        self.postprocess = postprocess
        self.codeswitch = self.lang == "zh_en"
        self.model_name = "conformer_talcs" if self.codeswitch else "conformer_u2pp_online_wenetspeech"
        self._executor = None

    def load(self) -> None:
        if self.loaded and (self.simulate or self._executor is not None):
            return
        start = time.perf_counter()
        self.load_error = ""
        self.runtime_note = ""
        self._executor = None

        try:
            if self.requested_simulate:
                self.simulate = True
                self.backend_name = "simulated"
                self.backend_detail = f"demo backend / {self.lang}"
                time.sleep(0.04)
            else:
                try:
                    paddle_module = importlib.import_module("paddle")
                    infer_module = importlib.import_module("paddlespeech.cli.asr.infer")
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装 PaddleSpeech 或 PaddlePaddle。请在 Python 3.11 环境中安装 `paddlespeech paddlepaddle` 后再使用真实模式。"
                    ) from exc

                target_device = "gpu" if self.device == "cuda" else "cpu"
                with contextlib.suppress(Exception):
                    if hasattr(paddle_module, "in_dynamic_mode") and not paddle_module.in_dynamic_mode():
                        paddle_module.disable_static()
                        self.runtime_note = "检测到 Paddle 处于静态图模式，已在真实评测前切回动态模式。"
                paddle_module.set_device(target_device)
                executor_cls = getattr(infer_module, "ASRExecutor")
                self._executor = executor_cls()
                self.simulate = False
                self.backend_name = "paddlespeech"
                self.backend_detail = f"{self.lang} / {target_device} / {self.model_name}"
                if self.codeswitch:
                    self.runtime_note = "zh_en 模式已自动启用 PaddleSpeech codeswitch，并切换到 conformer_talcs。"
                elif not self.runtime_note:
                    self.runtime_note = f"真实评测固定使用 PaddleSpeech 模型 {self.model_name}。"
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
        if self._executor is not None:  # pragma: no cover
            kwargs = {
                "audio_file": audio_path,
                "lang": self.lang,
                "codeswitch": self.codeswitch,
                "device": "gpu" if self.device == "cuda" else "cpu",
                "model": self.model_name,
            }
            result = self._executor(**kwargs)
            return str(result).strip()
        return build_simulated_prediction(audio_path, error_rate=0.09, seed_bias=41)

    def warmup(self, audio_path: str | None = None) -> None:
        if self.simulate or not audio_path or self.warmed_up:
            return
        start = time.perf_counter()
        self.transcribe(audio_path)
        self.load_time_ms = round(self.load_time_ms + (time.perf_counter() - start) * 1000, 3)
        self.warmed_up = True

    def unload(self) -> None:
        self._executor = None
        super().unload()
