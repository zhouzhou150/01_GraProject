from __future__ import annotations

import csv
import json
from datetime import datetime
from pathlib import Path

from asr_eval_system.schemas import AggregateReport


def export_report_bundle(report: AggregateReport, output_dir: str | Path) -> dict[str, str]:
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    stem = report.experiment_id
    json_path = target_dir / f"{stem}.json"
    csv_path = target_dir / f"{stem}_summary.csv"
    md_path = target_dir / f"{stem}.md"

    json_path.write_text(json.dumps(report.to_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
    _write_summary_csv(report, csv_path)
    md_path.write_text(_render_markdown_report(report), encoding="utf-8")
    return {
        "json": str(json_path),
        "csv": str(csv_path),
        "markdown": str(md_path),
        "created_at": datetime.now().isoformat(timespec="seconds"),
    }


def _write_summary_csv(report: AggregateReport, output_path: Path) -> None:
    if not report.summary:
        output_path.write_text("", encoding="utf-8")
        return
    headers = list(report.summary[0].keys())
    with output_path.open("w", encoding="utf-8-sig", newline="") as file:
        writer = csv.DictWriter(file, fieldnames=headers)
        writer.writeheader()
        writer.writerows(report.summary)


def _render_markdown_report(report: AggregateReport) -> str:
    lines = [
        f"# 实验报告：{report.experiment_id}",
        "",
        f"- 数据集：{report.dataset_name}",
        f"- 生成时间：{report.created_at}",
        "",
        "## 汇总结果",
        "",
        "| 模型 | CER | WER | SER | SemDist | 平均延迟(ms) | USS | 满意度等级 |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for item in report.summary:
        lines.append(
            f"| {item['model_id']} | {item['cer']:.4f} | {item['wer']:.4f} | {item['ser']:.4f} | "
            f"{item['semdist']:.2f} | {item['avg_latency_ms']:.2f} | {item['uss']:.2f} | {item['satisfaction_level']} |"
        )
    lines.extend(["", "## 结论", "", report.conclusion_text or "当前实验已完成。", ""])
    return "\n".join(lines)

