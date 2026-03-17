from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    def __init__(self, model_id: str, device: str = "cpu", simulate: bool = True) -> None:
        self.model_id = model_id
        self.device = device
        self.requested_simulate = simulate
        self.simulate = simulate
        self.backend_name = "simulated" if simulate else "real"
        self.backend_detail = ""
        self.runtime_note = ""
        self.loaded = False
        self.load_time_ms = 0.0
        self.load_error = ""
        self.warmed_up = False

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def warmup(self, audio_path: str | None = None) -> None:
        return None

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        self.loaded = False
        self.warmed_up = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "backend": self.backend_name,
            "backend_detail": self.backend_detail,
            "runtime_note": self.runtime_note,
            "loaded": self.loaded,
            "load_time_ms": self.load_time_ms,
            "requested_simulate": self.requested_simulate,
            "simulate": self.simulate,
            "load_error": self.load_error,
        }
