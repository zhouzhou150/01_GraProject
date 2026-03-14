from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class ModelAdapter(ABC):
    def __init__(self, model_id: str, device: str = "cpu", simulate: bool = True) -> None:
        self.model_id = model_id
        self.device = device
        self.simulate = simulate
        self.backend_name = "simulated" if simulate else "real"
        self.loaded = False
        self.load_time_ms = 0.0

    @abstractmethod
    def load(self) -> None:
        raise NotImplementedError

    def warmup(self) -> None:
        return None

    @abstractmethod
    def transcribe(self, audio_path: str) -> str:
        raise NotImplementedError

    def unload(self) -> None:
        self.loaded = False

    def metadata(self) -> dict[str, Any]:
        return {
            "model_id": self.model_id,
            "device": self.device,
            "backend": self.backend_name,
            "loaded": self.loaded,
            "load_time_ms": self.load_time_ms,
        }

