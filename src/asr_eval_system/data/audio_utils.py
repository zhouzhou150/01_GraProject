from __future__ import annotations

import contextlib
import wave
from pathlib import Path


def read_wave_duration(audio_path: str | Path) -> float:
    path = Path(audio_path)
    with contextlib.closing(wave.open(str(path), "rb")) as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            return 0.0
        return round(frames / frame_rate, 4)


def find_sidecar_transcript(audio_path: str | Path) -> str | None:
    path = Path(audio_path)
    for suffix in (".txt", ".lab"):
        sidecar = path.with_suffix(suffix)
        if sidecar.exists():
            return sidecar.read_text(encoding="utf-8").strip()
    return None

