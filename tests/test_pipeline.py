import gc
import json
import sys
import tempfile
import unittest
import wave
from pathlib import Path
from unittest.mock import Mock

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_eval_system.metrics.satisfaction import build_satisfaction_profile
from asr_eval_system.models.faster_whisper_adapter import FasterWhisperAdapter
from asr_eval_system.models.paddlespeech_adapter import PaddleSpeechAdapter
from asr_eval_system.models.registry import build_model_registry
from asr_eval_system.reporting.report_generator import export_report_bundle
from asr_eval_system.runner.evaluation import run_experiment, run_experiment_from_specs
from asr_eval_system.schemas import DatasetManifest, ExperimentConfig
from asr_eval_system.storage.database import DatabaseManager


class PipelineTests(unittest.TestCase):
    def setUp(self) -> None:
        self.profile = build_satisfaction_profile(
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

    def test_end_to_end_pipeline(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            items: list[DatasetManifest] = []
            for index, text in enumerate(["深度学习语音识别", "语音测试系统"], start=1):
                wav_path = root / f"sample_{index}.wav"
                txt_path = wav_path.with_suffix(".txt")
                with wave.open(str(wav_path), "wb") as wav_file:
                    wav_file.setnchannels(1)
                    wav_file.setsampwidth(2)
                    wav_file.setframerate(16000)
                    wav_file.writeframes(b"\x00\x00" * 16000)
                txt_path.write_text(text, encoding="utf-8")
                items.append(
                    DatasetManifest(
                        sample_id=f"s{index}",
                        audio_path=str(wav_path),
                        transcript=text,
                        duration_sec=1.0,
                        scene_tag="quiet" if index == 1 else "noise_light",
                        noise_tag="none" if index == 1 else "light",
                    )
                )

            config = ExperimentConfig(
                experiment_id="exp_demo",
                model_ids=["cnn_ctc", "rnn_ctc", "faster_whisper", "paddlespeech"],
                dataset_name="demo_set",
            )
            report = run_experiment(config, items, build_model_registry(simulate=True), self.profile)
            self.assertEqual(len(report.summary), 4)
            self.assertEqual(len(report.sample_results), 8)
            self.assertIn("uss_ranking", report.charts)
            self.assertTrue(all(item["runtime_mode"] == "模拟" for item in report.summary))
            self.assertTrue(all(item["backend"] for item in report.summary))

            export_paths = export_report_bundle(report, root)
            self.assertTrue(Path(export_paths["json"]).exists())
            self.assertTrue(Path(export_paths["csv"]).exists())
            self.assertTrue(Path(export_paths["markdown"]).exists())

            database = DatabaseManager(root / "runtime.db")
            database.save_experiment(report)
            loaded = database.get_report("exp_demo")
            self.assertIsNotNone(loaded)
            self.assertEqual(len(loaded.summary), 4)
            experiments = database.list_experiments()
            self.assertEqual(experiments[0]["experiment_id"], "exp_demo")
            del loaded
            del database
            gc.collect()

            payload = json.loads(Path(export_paths["json"]).read_text(encoding="utf-8"))
            self.assertEqual(payload["experiment_id"], "exp_demo")

    def test_run_experiment_from_specs_supports_simulated_specs(self) -> None:
        with tempfile.TemporaryDirectory() as tmp_dir:
            root = Path(tmp_dir)
            wav_path = root / "sample.wav"
            with wave.open(str(wav_path), "wb") as wav_file:
                wav_file.setnchannels(1)
                wav_file.setsampwidth(2)
                wav_file.setframerate(16000)
                wav_file.writeframes(b"\x00\x00" * 16000)
            items = [
                DatasetManifest(
                    sample_id="s1",
                    audio_path=str(wav_path),
                    transcript="测试样本",
                    duration_sec=1.0,
                )
            ]
            config = ExperimentConfig(
                experiment_id="exp_specs",
                model_ids=["cnn_ctc", "faster_whisper"],
                dataset_name="demo_set",
            )
            report = run_experiment_from_specs(
                config=config,
                dataset_items=items,
                model_specs=[
                    {"model_id": "cnn_ctc", "device": "cpu", "simulate": True, "options": {}},
                    {
                        "model_id": "faster_whisper",
                        "device": "cpu",
                        "simulate": True,
                        "options": {"model_size": "tiny", "compute_type": "int8", "lang": "zh"},
                    },
                ],
                profile=self.profile,
            )
            self.assertEqual(len(report.summary), 2)
            self.assertEqual(len(report.sample_results), 2)

    def test_faster_whisper_normalizes_unstable_compute_types(self) -> None:
        gpu_adapter = FasterWhisperAdapter(device="cuda", simulate=False, compute_type="float32")
        self.assertEqual(gpu_adapter.compute_type, "float16")
        self.assertTrue(gpu_adapter.runtime_note)

        cpu_adapter = FasterWhisperAdapter(device="cpu", simulate=False, compute_type="float16")
        self.assertEqual(cpu_adapter.compute_type, "int8")
        self.assertTrue(cpu_adapter.runtime_note)

    def test_paddlespeech_zh_en_enables_codeswitch(self) -> None:
        adapter = PaddleSpeechAdapter(device="cpu", simulate=False, lang="zh_en")
        adapter.loaded = True
        adapter.simulate = False
        adapter._executor = Mock(return_value="test result")
        text = adapter.transcribe("demo.wav")
        self.assertEqual(text, "test result")
        adapter._executor.assert_called_once_with(
            audio_file="demo.wav",
            lang="zh_en",
            codeswitch=True,
            device="cpu",
            model="conformer_talcs",
        )

    def test_paddlespeech_zh_uses_explicit_default_model(self) -> None:
        adapter = PaddleSpeechAdapter(device="cpu", simulate=False, lang="zh")
        adapter.loaded = True
        adapter.simulate = False
        adapter._executor = Mock(return_value="test result")
        text = adapter.transcribe("demo.wav")
        self.assertEqual(text, "test result")
        adapter._executor.assert_called_once_with(
            audio_file="demo.wav",
            lang="zh",
            codeswitch=False,
            device="cpu",
            model="conformer_u2pp_online_wenetspeech",
        )


if __name__ == "__main__":
    unittest.main()
