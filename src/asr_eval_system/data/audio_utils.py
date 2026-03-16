from __future__ import annotations

import contextlib
import re
import wave
from pathlib import Path

import numpy as np


SUPPORTED_AUDIO_EXTENSIONS = (".wav", ".mp3", ".flac", ".m4a", ".ogg", ".opus", ".aac", ".wma")
SUPPORTED_TRANSCRIPT_EXTENSIONS = (".txt", ".lab", ".trn")


def normalize_uploaded_name(name: str) -> str:
    return name.replace("\\", "/").strip("/")


def extract_transcript_text(content: str, suffix: str = "") -> str:
    lines = [line.strip() for line in content.replace("\r\n", "\n").splitlines() if line.strip()]
    if suffix.lower() == ".trn":
        return lines[0] if lines else ""
    return " ".join(lines).strip()


def decode_transcript_bytes(raw: bytes, suffix: str = "") -> str:
    for encoding in ("utf-8", "utf-8-sig", "gb18030", "gbk"):
        try:
            return extract_transcript_text(raw.decode(encoding), suffix=suffix)
        except UnicodeDecodeError:
            continue
    return extract_transcript_text(raw.decode("utf-8", errors="ignore"), suffix=suffix)


def transcript_match_keys(file_name: str) -> set[str]:
    normalized = normalize_uploaded_name(file_name)
    lowered = normalized.lower()
    base_name = normalized
    for suffix in SUPPORTED_TRANSCRIPT_EXTENSIONS:
        if lowered.endswith(suffix):
            base_name = normalized[: -len(suffix)]
            break

    path = Path(base_name)
    keys = {normalize_uploaded_name(base_name), path.name}
    if path.stem:
        keys.add(path.stem)
    return {key for key in keys if key}


def audio_match_keys(file_name: str) -> tuple[str, ...]:
    normalized = normalize_uploaded_name(file_name)
    path = Path(normalized)
    keys = [normalized, path.name]
    if path.stem:
        keys.append(path.stem)
    deduped: list[str] = []
    for key in keys:
        if key and key not in deduped:
            deduped.append(key)
    return tuple(deduped)


def resolve_transcript_text(transcript_map: dict[str, str], audio_name: str) -> str:
    for key in audio_match_keys(audio_name):
        if key in transcript_map:
            return transcript_map[key]
    return ""


def build_sample_id(source_name: str, fallback_index: int) -> str:
    normalized = normalize_uploaded_name(source_name)
    stem = re.sub(r"\.[^.]+$", "", normalized)
    sanitized = re.sub(r"[^0-9A-Za-z\u4e00-\u9fff]+", "_", stem).strip("_")
    return sanitized or f"sample_{fallback_index:04d}"


def _read_duration_from_wave(audio_path: Path) -> float:
    with contextlib.closing(wave.open(str(audio_path), "rb")) as wav_file:
        frames = wav_file.getnframes()
        frame_rate = wav_file.getframerate()
        if frame_rate <= 0:
            return 0.0
        return round(frames / frame_rate, 4)


def read_audio_duration(audio_path: str | Path) -> float:
    path = Path(audio_path)
    if not path.exists():
        return 0.0

    if path.suffix.lower() == ".wav":
        try:
            return _read_duration_from_wave(path)
        except (wave.Error, EOFError):
            pass

    try:
        import soundfile as sf
    except ImportError:  # pragma: no cover
        sf = None
    if sf is not None:
        try:
            info = sf.info(str(path))
        except RuntimeError:
            pass
        else:
            if info.samplerate > 0:
                return round(info.frames / info.samplerate, 4)

    try:
        import torchaudio
    except ImportError:  # pragma: no cover
        torchaudio = None
    if torchaudio is not None:
        try:
            info = torchaudio.info(str(path))
        except (RuntimeError, OSError):
            return 0.0
        if info.sample_rate > 0:
            return round(info.num_frames / info.sample_rate, 4)

    return 0.0


def read_wave_duration(audio_path: str | Path) -> float:
    return read_audio_duration(audio_path)


def _load_audio_samples(audio_path: str | Path) -> tuple[np.ndarray, int]:
    path = Path(audio_path)
    try:
        import torchaudio
    except ImportError:  # pragma: no cover
        torchaudio = None
    if torchaudio is not None:
        try:
            waveform, sample_rate = torchaudio.load(str(path))
        except (RuntimeError, OSError):
            waveform = None
        else:
            return waveform.transpose(0, 1).contiguous().cpu().numpy(), int(sample_rate)

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("当前环境缺少 soundfile，无法读取多格式音频。") from exc

    samples, sample_rate = sf.read(str(path), always_2d=True, dtype="float32")
    return np.asarray(samples, dtype=np.float32), int(sample_rate)


def _resample_samples(samples: np.ndarray, original_rate: int, target_rate: int) -> np.ndarray:
    if original_rate == target_rate:
        return samples.astype(np.float32, copy=False)

    try:
        from scipy.signal import resample_poly
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("当前环境缺少 scipy，无法重采样音频。") from exc

    resampled = resample_poly(samples, target_rate, original_rate)
    return np.asarray(resampled, dtype=np.float32)


def normalize_audio_file(
    source_path: str | Path,
    destination_path: str | Path,
    target_sample_rate: int = 16000,
) -> float:
    samples, sample_rate = _load_audio_samples(source_path)
    if samples.ndim == 2:
        mono = samples.mean(axis=1)
    else:
        mono = samples
    mono = np.asarray(mono, dtype=np.float32)
    mono = _resample_samples(mono, sample_rate, target_sample_rate)
    mono = np.clip(mono, -1.0, 1.0)

    try:
        import soundfile as sf
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("当前环境缺少 soundfile，无法写入规范化后的 WAV 文件。") from exc

    output_path = Path(destination_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), mono, target_sample_rate, subtype="PCM_16")
    return round(len(mono) / target_sample_rate, 4)


def audio_player_format(audio_path: str | Path) -> str:
    suffix = Path(audio_path).suffix.lower()
    if suffix == ".mp3":
        return "audio/mp3"
    if suffix == ".flac":
        return "audio/flac"
    if suffix in {".m4a", ".aac"}:
        return "audio/mp4"
    if suffix in {".ogg", ".opus"}:
        return "audio/ogg"
    return "audio/wav"


def find_sidecar_transcript(audio_path: str | Path) -> str | None:
    path = Path(audio_path)
    candidates = [
        path.with_suffix(".txt"),
        path.with_suffix(".lab"),
        path.with_suffix(".trn"),
        Path(f"{path}.trn"),
    ]
    for sidecar in candidates:
        if sidecar.exists():
            return decode_transcript_bytes(sidecar.read_bytes(), suffix=sidecar.suffix)
    return None
