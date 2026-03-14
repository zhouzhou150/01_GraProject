from __future__ import annotations

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from asr_eval_system.service import run_default_experiment


def main() -> None:
    report, exports = run_default_experiment(simulate=True)
    print("实验完成：", report.experiment_id)
    for item in report.summary:
        print(
            f"{item['model_id']}: CER={item['cer']:.4f}, "
            f"AvgLatency={item['avg_latency_ms']:.2f}ms, USS={item['uss']:.2f}, "
            f"Level={item['satisfaction_level']}"
        )
    print("导出文件：")
    for key, value in exports.items():
        if key != "created_at":
            print(f"  {key}: {value}")


if __name__ == "__main__":
    main()

