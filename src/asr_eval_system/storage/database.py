from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path

from asr_eval_system.schemas import AggregateReport


class DatabaseManager:
    def __init__(self, database_path: str | Path) -> None:
        self.database_path = Path(database_path)
        self.database_path.parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        return sqlite3.connect(self.database_path)

    def _initialize(self) -> None:
        with closing(self._connect()) as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS experiments (
                    experiment_id TEXT PRIMARY KEY,
                    dataset_name TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    config_json TEXT NOT NULL,
                    satisfaction_profile_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS sample_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    sample_id TEXT NOT NULL,
                    status TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS aggregate_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    model_id TEXT NOT NULL,
                    payload_json TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS export_records (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    experiment_id TEXT NOT NULL,
                    export_type TEXT NOT NULL,
                    export_path TEXT NOT NULL,
                    created_at TEXT NOT NULL
                );
                """
            )
            conn.commit()

    def save_experiment(self, report: AggregateReport) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT OR REPLACE INTO experiments (
                    experiment_id, dataset_name, created_at, config_json, satisfaction_profile_json
                )
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    report.experiment_id,
                    report.dataset_name,
                    report.created_at,
                    json.dumps(report.config, ensure_ascii=False),
                    json.dumps(report.satisfaction_profile, ensure_ascii=False),
                ),
            )
            conn.execute("DELETE FROM sample_results WHERE experiment_id = ?", (report.experiment_id,))
            conn.execute("DELETE FROM aggregate_results WHERE experiment_id = ?", (report.experiment_id,))
            for item in report.sample_results:
                conn.execute(
                    """
                    INSERT INTO sample_results (experiment_id, model_id, sample_id, status, payload_json)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    (
                        report.experiment_id,
                        item["model_id"],
                        item["sample_id"],
                        item["status"],
                        json.dumps(item, ensure_ascii=False),
                    ),
                )
            conn.commit()
            for item in report.summary:
                conn.execute(
                    """
                    INSERT INTO aggregate_results (experiment_id, model_id, payload_json)
                    VALUES (?, ?, ?)
                    """,
                    (
                        report.experiment_id,
                        item["model_id"],
                        json.dumps(item, ensure_ascii=False),
                    ),
                )
            conn.commit()

    def record_export(self, experiment_id: str, export_type: str, export_path: str, created_at: str) -> None:
        with closing(self._connect()) as conn:
            conn.execute(
                """
                INSERT INTO export_records (experiment_id, export_type, export_path, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (experiment_id, export_type, export_path, created_at),
            )
            conn.commit()

    def list_experiments(self) -> list[dict[str, str]]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT experiment_id, dataset_name, created_at FROM experiments ORDER BY created_at DESC"
            ).fetchall()
        return [{"experiment_id": row[0], "dataset_name": row[1], "created_at": row[2]} for row in rows]

    def get_report(self, experiment_id: str) -> AggregateReport | None:
        with closing(self._connect()) as conn:
            header = conn.execute(
                """
                SELECT dataset_name, created_at, config_json, satisfaction_profile_json
                FROM experiments WHERE experiment_id = ?
                """,
                (experiment_id,),
            ).fetchone()
            if header is None:
                return None
            summary_rows = conn.execute(
                "SELECT payload_json FROM aggregate_results WHERE experiment_id = ? ORDER BY model_id",
                (experiment_id,),
            ).fetchall()
            sample_rows = conn.execute(
                "SELECT payload_json FROM sample_results WHERE experiment_id = ? ORDER BY model_id, sample_id",
                (experiment_id,),
            ).fetchall()

        return AggregateReport(
            experiment_id=experiment_id,
            dataset_name=header[0],
            created_at=header[1],
            config=json.loads(header[2]),
            summary=[json.loads(row[0]) for row in summary_rows],
            sample_results=[json.loads(row[0]) for row in sample_rows],
            satisfaction_profile=json.loads(header[3]),
            charts={},
            conclusion_text="",
        )
