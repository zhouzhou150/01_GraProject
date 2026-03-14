from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SAMPLE_DIR = ROOT / "data" / "sample"
MANIFEST_DIR = ROOT / "data" / "manifests"


def generate_wave(path: Path, frequency: float, seconds: float = 1.0, sample_rate: int = 16000) -> None:
    frames = []
    for index in range(int(sample_rate * seconds)):
        value = int(8000 * math.sin(2 * math.pi * frequency * index / sample_rate))
        frames.append(struct.pack("<h", value))
    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(b"".join(frames))


def main() -> None:
    SAMPLE_DIR.mkdir(parents=True, exist_ok=True)
    MANIFEST_DIR.mkdir(parents=True, exist_ok=True)

    specs = [
        ("sample_01", "深度学习语音识别", 440.0, "quiet", "none"),
        ("sample_02", "语音识别性能测试", 554.37, "noise_light", "light"),
        ("sample_03", "本科毕业设计系统", 659.25, "noise_mid", "medium"),
    ]

    manifest = []
    for sample_id, text, frequency, scene_tag, noise_tag in specs:
        wav_path = SAMPLE_DIR / f"{sample_id}.wav"
        txt_path = SAMPLE_DIR / f"{sample_id}.txt"
        generate_wave(wav_path, frequency=frequency)
        txt_path.write_text(text, encoding="utf-8")
        manifest.append(
            {
                "sample_id": sample_id,
                "audio_path": str(wav_path),
                "transcript": text,
                "duration_sec": 1.0,
                "split": "test",
                "scene_tag": scene_tag,
                "noise_tag": noise_tag,
                "accent_tag": "standard",
            }
        )

    (MANIFEST_DIR / "demo_manifest.json").write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print("示例数据集已生成：", MANIFEST_DIR / "demo_manifest.json")


if __name__ == "__main__":
    main()

