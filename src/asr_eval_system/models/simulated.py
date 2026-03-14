from __future__ import annotations

import random
from pathlib import Path

from asr_eval_system.data.audio_utils import find_sidecar_transcript


def build_simulated_prediction(audio_path: str, error_rate: float, seed_bias: int = 0) -> str:
    base_text = find_sidecar_transcript(audio_path)
    if not base_text:
        return Path(audio_path).stem.replace("_", "")

    chars = list(base_text)
    seed = sum(ord(char) for char in Path(audio_path).stem) + seed_bias
    rng = random.Random(seed)
    operations = max(1, round(len(chars) * error_rate)) if chars else 0
    for _ in range(operations):
        if not chars:
            break
        action = rng.choice(["drop", "swap", "repeat"])
        index = rng.randrange(len(chars))
        if action == "drop" and len(chars) > 1:
            chars.pop(index)
        elif action == "swap":
            chars[index] = rng.choice(chars)
        else:
            chars.insert(index, chars[index])
    return "".join(chars)

