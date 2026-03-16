import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_eval_system.data.audio_utils import build_sample_id, decode_transcript_bytes, resolve_transcript_text, transcript_match_keys
from asr_eval_system.data.dataset import load_manifest, validate_manifest
from asr_eval_system.metrics.satisfaction import build_satisfaction_profile, compute_uss
from asr_eval_system.metrics.text_metrics import cer, semdist_score, ser, wer
from asr_eval_system.workflow import compute_workflow_progress


class MetricsAndDataTests(unittest.TestCase):
    def test_text_metrics(self) -> None:
        self.assertEqual(cer("你好", "你好"), 0.0)
        self.assertGreater(cer("你好世界", "你好"), 0.0)
        self.assertEqual(ser("hello", "hello"), 0.0)
        self.assertEqual(wer("ni hao", "ni hao"), 0.0)
        self.assertGreater(semdist_score("深度学习语音识别", "深度语音识别"), 50.0)

    def test_satisfaction_monotonicity(self) -> None:
        profile = build_satisfaction_profile(
            {
                "lit_weights": {
                    "accuracy_semantic": 0.45,
                    "latency": 0.25,
                    "robustness": 0.20,
                    "resource": 0.10,
                },
                "good_bad_thresholds": {"high": 85, "good": 70, "fair": 60},
            }
        )
        better, _, _ = compute_uss(0.10, 90, 500, 0.4, 90, 25, 512, 0, 120, profile)
        worse, _, _ = compute_uss(0.30, 70, 1800, 1.2, 70, 65, 2048, 0, 1800, profile)
        self.assertGreater(better, worse)

    def test_manifest_load_and_validate(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "demo.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)

            manifest_path = root / "manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "sample_id": "s1",
                            "audio_path": str(wav_path),
                            "transcript": "你好",
                            "duration_sec": 0,
                        }
                    ],
                    ensure_ascii=False,
                ),
                encoding="utf-8",
            )
            items = load_manifest(manifest_path)
            issues = validate_manifest(items)
            self.assertEqual(len(items), 1)
            self.assertEqual(issues, [])
            self.assertGreater(items[0].duration_sec, 0)

    def test_transcript_matching_supports_trn(self) -> None:
        transcript_map: dict[str, str] = {}
        for key in transcript_match_keys("folder/A2_1.wav.trn"):
            transcript_map[key] = decode_transcript_bytes("你好 世界\nmetadata".encode("utf-8"), suffix=".trn")
        self.assertEqual(resolve_transcript_text(transcript_map, "folder/A2_1.wav"), "你好 世界")
        self.assertEqual(build_sample_id("folder/A2_1.wav", 1), "folder_A2_1")

    def test_workflow_progress_is_sequential(self) -> None:
        progress = compute_workflow_progress(dataset_ready=False, loaded_model_count=2, performance_ready=True, overall_ready=True)
        self.assertEqual(progress.progress_value, 0.0)
        self.assertEqual(progress.current_step, "导入音频与参考文本")

        progress = compute_workflow_progress(dataset_ready=True, loaded_model_count=1, performance_ready=False, overall_ready=True)
        self.assertEqual(progress.progress_value, 0.5)
        self.assertEqual(progress.current_step, "运行性能测试与总体测试")


if __name__ == "__main__":
    unittest.main()
