from __future__ import annotations

import contextlib
from importlib.util import find_spec
import os
from pathlib import Path
import sys
import time

from asr_eval_system.models.base import ModelAdapter
from asr_eval_system.models.simulated import build_simulated_prediction


_DLL_DIRECTORY_HANDLES: list[object] = []


def _normalize_compute_type(device: str, compute_type: str) -> tuple[str, str]:
    requested = (compute_type or "int8").lower()
    if device == "cuda" and requested == "float32":
        return "float16", "CUDA 下 float32 容易触发 Faster-Whisper/CT2 不稳定，已自动切换为 float16。"
    if device == "cpu" and requested == "float16":
        return "int8", "CPU 下 float16 兼容性较弱，已自动切换为 int8。"
    return requested, ""


def _ctranslate2_package_dir() -> Path | None:
    spec = find_spec("ctranslate2")
    if spec is None or not spec.origin:
        return None
    return Path(spec.origin).resolve().parent


def _candidate_runtime_paths() -> list[Path]:
    python_root = Path(sys.executable).resolve().parent
    candidates: list[Path] = [
        python_root,
        python_root / "Library" / "bin",
        python_root / "Library" / "usr" / "bin",
        python_root / "Scripts",
    ]

    ct2_dir = _ctranslate2_package_dir()
    if ct2_dir is not None:
        candidates.append(ct2_dir)
        candidates.extend(
            [
                ct2_dir.parent / "nvidia" / "cublas" / "bin",
                ct2_dir.parent / "nvidia" / "cudnn" / "bin",
                ct2_dir.parent / "nvidia" / "cuda_runtime" / "bin",
            ]
        )

    cuda_path = os.environ.get("CUDA_PATH")
    if cuda_path:
        cuda_root = Path(cuda_path)
        candidates.extend([cuda_root / "bin", cuda_root / "libnvvp"])

    deduped: list[Path] = []
    seen: set[str] = set()
    for path in candidates:
        normalized = str(path).strip()
        if not normalized or normalized in seen or not path.exists():
            continue
        seen.add(normalized)
        deduped.append(path)
    return deduped


def _prepare_windows_runtime_paths() -> None:
    if os.name != "nt":
        return

    candidates = _candidate_runtime_paths()
    existing_entries = os.environ.get("PATH", "").split(os.pathsep)
    merged_entries: list[str] = []
    for entry in [str(path) for path in candidates] + existing_entries:
        normalized = entry.strip()
        if normalized and normalized not in merged_entries:
            merged_entries.append(normalized)
    os.environ["PATH"] = os.pathsep.join(merged_entries)

    if hasattr(os, "add_dll_directory"):
        for candidate in candidates:
            with contextlib.suppress(OSError, FileNotFoundError):
                _DLL_DIRECTORY_HANDLES.append(os.add_dll_directory(str(candidate)))


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
        self.requested_compute_type = (compute_type or "int8").lower()
        self.compute_type, self.runtime_note = _normalize_compute_type(device, self.requested_compute_type)
        self.language = language
        self._model = None

    def load(self) -> None:
        if self.loaded and (self.simulate or self._model is not None):
            return
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
                _prepare_windows_runtime_paths()
                try:
                    from faster_whisper import WhisperModel
                except ImportError as exc:
                    raise RuntimeError(
                        "未安装 faster-whisper。请在 Python 3.11 环境中安装 `faster-whisper` 后再使用真实模式。"
                    ) from exc
                except OSError as exc:
                    raise RuntimeError(
                        "Faster-Whisper 依赖的 Windows DLL 未加载成功。"
                        f" 原始错误：{exc}。当前 Python: {sys.executable}。"
                        " 请优先使用 `python -m streamlit run app/streamlit_app.py` 从 Python 3.11 环境启动，"
                        "并确认 CUDA 与 ctranslate2 依赖已加入系统路径。"
                    ) from exc

                try:
                    self._model = WhisperModel(
                        self.model_size,
                        device=self.device,
                        compute_type=self.compute_type,
                    )
                except OSError as exc:
                    raise RuntimeError(
                        "Faster-Whisper 模型初始化失败，通常是 CUDA/ct2 运行时库未被当前进程找到。"
                        f" 原始错误：{exc}。当前 Python: {sys.executable}。"
                    ) from exc

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

    def warmup(self, audio_path: str | None = None) -> None:
        if self.simulate or not audio_path or self.warmed_up:
            return
        start = time.perf_counter()
        self.transcribe(audio_path)
        self.load_time_ms = round(self.load_time_ms + (time.perf_counter() - start) * 1000, 3)
        self.warmed_up = True

    def unload(self) -> None:
        self._model = None
        super().unload()
