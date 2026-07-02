from __future__ import annotations

import sqlite3
import subprocess
from pathlib import Path
from typing import Any

import pandas as pd


def _connect(db_path: str | Path) -> sqlite3.Connection:
    return sqlite3.connect(Path(db_path))


def _current_git_commit() -> str | None:
    try:
        repo_root = Path(__file__).resolve().parents[2]
        result = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
        )
    except (OSError, subprocess.CalledProcessError):
        return None

    commit = result.stdout.strip()
    return commit or None


def ensure_schema(db_path: str | Path) -> None:
    Path(db_path).parent.mkdir(parents=True, exist_ok=True)
    with _connect(db_path) as conn:
        conn.executescript(
            """
            CREATE TABLE IF NOT EXISTS runs (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                random_state    INTEGER NOT NULL DEFAULT 42,
                train_catboost  INTEGER NOT NULL DEFAULT 0,
                run_tabpfn      INTEGER NOT NULL DEFAULT 0,
                run_tabfm       INTEGER NOT NULL DEFAULT 0,
                run_autogluon   INTEGER NOT NULL DEFAULT 0,
                autogluon_tl    INTEGER NOT NULL DEFAULT 600,
                output_dir      TEXT,
                duration_sec    REAL,
                git_commit      TEXT
            );

            CREATE TABLE IF NOT EXISTS model_metrics (
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                model_name  TEXT NOT NULL,
                mae_eur     REAL NOT NULL,
                median_ae   REAL NOT NULL,
                mape        REAL NOT NULL,
                within_10   REAL NOT NULL,
                within_15   REAL NOT NULL,
                UNIQUE(run_id, model_name)
            );

            CREATE TABLE IF NOT EXISTS skipped_models (
                run_id      INTEGER NOT NULL REFERENCES runs(id),
                model_name  TEXT NOT NULL,
                reason      TEXT NOT NULL,
                UNIQUE(run_id, model_name)
            );
            """
        )
        columns = {row[1] for row in conn.execute("PRAGMA table_info(runs)").fetchall()}
        if "run_tabfm" not in columns:
            conn.execute("ALTER TABLE runs ADD COLUMN run_tabfm INTEGER NOT NULL DEFAULT 0")


def insert_run(
    db_path: str | Path,
    *,
    random_state: int,
    train_catboost: bool,
    run_tabpfn: bool,
    run_tabfm: bool,
    run_autogluon: bool,
    autogluon_tl: int,
    output_dir: str | Path | None,
    duration_sec: float | None,
    git_commit: str | None = None,
) -> int:
    ensure_schema(db_path)
    commit = _current_git_commit() if git_commit is None else git_commit
    with _connect(db_path) as conn:
        cursor = conn.execute(
            """
            INSERT INTO runs (
                random_state, train_catboost, run_tabpfn, run_tabfm, run_autogluon,
                autogluon_tl, output_dir, duration_sec, git_commit
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                random_state,
                int(train_catboost),
                int(run_tabpfn),
                int(run_tabfm),
                int(run_autogluon),
                autogluon_tl,
                None if output_dir is None else str(output_dir),
                duration_sec,
                commit,
            ),
        )
        lastrowid = cursor.lastrowid
        if lastrowid is None:
            raise RuntimeError("failed to insert run row")
        return int(lastrowid)


def insert_metrics(db_path: str | Path, run_id: int, metrics: dict[str, dict[str, Any]]) -> None:
    ensure_schema(db_path)
    rows = [
        (
            run_id,
            model_name,
            values["mae_eur"],
            values["median_ae"],
            values["mape"],
            values["within_10"],
            values["within_15"],
        )
        for model_name, values in metrics.items()
    ]
    with _connect(db_path) as conn:
        conn.executemany(
            """
            INSERT OR IGNORE INTO model_metrics (
                run_id, model_name, mae_eur, median_ae, mape, within_10, within_15
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            rows,
        )


def insert_skipped(db_path: str | Path, run_id: int, skipped: dict[str, str]) -> None:
    ensure_schema(db_path)
    rows = [(run_id, model_name, reason) for model_name, reason in skipped.items()]
    with _connect(db_path) as conn:
        conn.executemany(
            "INSERT OR IGNORE INTO skipped_models (run_id, model_name, reason) VALUES (?, ?, ?)",
            rows,
        )


def get_latest_run(db_path: str | Path) -> dict[str, Any] | None:
    ensure_schema(db_path)
    with _connect(db_path) as conn:
        run_row = conn.execute(
            """
            SELECT id, created_at, random_state, train_catboost, run_tabpfn, run_tabfm,
                   run_autogluon, autogluon_tl, output_dir, duration_sec, git_commit
            FROM runs
            ORDER BY id DESC
            LIMIT 1
            """
        ).fetchone()
        if run_row is None:
            return None

        columns = [
            "id",
            "created_at",
            "random_state",
            "train_catboost",
            "run_tabpfn",
            "run_tabfm",
            "run_autogluon",
            "autogluon_tl",
            "output_dir",
            "duration_sec",
            "git_commit",
        ]
        run_data = dict(zip(columns, run_row))
        metrics_rows = conn.execute(
            """
            SELECT model_name, mae_eur, median_ae, mape, within_10, within_15
            FROM model_metrics
            WHERE run_id = ?
            ORDER BY model_name
            """,
            (run_data["id"],),
        ).fetchall()
        skipped_rows = conn.execute(
            """
            SELECT model_name, reason
            FROM skipped_models
            WHERE run_id = ?
            ORDER BY model_name
            """,
            (run_data["id"],),
        ).fetchall()

    run_data["metrics"] = {
        row[0]: {
            "mae_eur": row[1],
            "median_ae": row[2],
            "mape": row[3],
            "within_10": row[4],
            "within_15": row[5],
        }
        for row in metrics_rows
    }
    run_data["skipped"] = {row[0]: row[1] for row in skipped_rows}
    return run_data


def get_run_history(db_path: str | Path) -> pd.DataFrame:
    ensure_schema(db_path)
    query = """
        SELECT
            r.id,
            r.created_at,
            r.random_state,
            r.train_catboost,
            r.run_tabpfn,
            r.run_tabfm,
            r.run_autogluon,
            r.autogluon_tl,
            r.duration_sec,
            COUNT(m.model_name) AS model_count
        FROM runs AS r
        LEFT JOIN model_metrics AS m ON m.run_id = r.id
        GROUP BY r.id, r.created_at, r.random_state, r.train_catboost,
                 r.run_tabpfn, r.run_tabfm, r.run_autogluon, r.autogluon_tl, r.duration_sec
        ORDER BY r.id
    """
    with _connect(db_path) as conn:
        return pd.read_sql_query(query, conn)
