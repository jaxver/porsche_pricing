from __future__ import annotations

import sqlite3

import pandas as pd

from elferspot_listings.modeling import benchmark_db


def _db_path(tmp_path):
    return tmp_path / "benchmark_runs.db"


def _sample_metrics():
    return {
        "ridge": {
            "mae_eur": 1111.0,
            "median_ae": 900.0,
            "mape": 0.12,
            "within_10": 0.31,
            "within_15": 0.44,
        },
        "catboost": {
            "mae_eur": 987.0,
            "median_ae": 750.0,
            "mape": 0.09,
            "within_10": 0.41,
            "within_15": 0.58,
        },
    }


def test_ensure_schema_creates_tables(tmp_path):
    db_path = _db_path(tmp_path)

    benchmark_db.ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT name FROM sqlite_master WHERE type='table' ORDER BY name").fetchall()

    assert {row[0] for row in rows} >= {"runs", "model_metrics", "skipped_models"}


def test_ensure_schema_is_idempotent(tmp_path):
    db_path = _db_path(tmp_path)

    benchmark_db.ensure_schema(db_path)
    benchmark_db.ensure_schema(db_path)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM sqlite_master WHERE type='table'").fetchone()[0]

    assert count >= 3


def test_ensure_schema_creates_parent_directory(tmp_path):
    db_path = tmp_path / "nested" / "benchmark_runs.db"

    benchmark_db.ensure_schema(db_path)

    assert db_path.exists()
    assert db_path.parent.exists()


def test_insert_run_returns_id(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    monkeypatch.setattr(benchmark_db, "_current_git_commit", lambda: "abc123")

    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=True,
        autogluon_tl=600,
        output_dir=tmp_path / "out",
        duration_sec=12.5,
    )

    assert isinstance(run_id, int)
    assert run_id == 1


def test_insert_run_stores_all_columns(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    monkeypatch.setattr(benchmark_db, "_current_git_commit", lambda: "abc123")

    run_id = benchmark_db.insert_run(
        db_path,
        random_state=7,
        train_catboost=False,
        run_tabpfn=True,
        run_tabfm=True,
        run_autogluon=True,
        autogluon_tl=1200,
        output_dir=tmp_path / "results" / "run-1",
        duration_sec=99.25,
    )

    with sqlite3.connect(db_path) as conn:
        row = conn.execute(
            """
            SELECT random_state, train_catboost, run_tabpfn, run_tabfm, run_autogluon,
                   autogluon_tl, output_dir, duration_sec, git_commit
            FROM runs
            WHERE id = ?
            """,
            (run_id,),
        ).fetchone()

    assert row == (7, 0, 1, 1, 1, 1200, str(tmp_path / "results" / "run-1"), 99.25, "abc123")


def test_insert_metrics_stores_all_models(tmp_path):
    db_path = _db_path(tmp_path)
    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
        git_commit="abc123",
    )

    benchmark_db.insert_metrics(db_path, run_id, _sample_metrics())

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute(
            "SELECT model_name, mae_eur, median_ae, mape, within_10, within_15 FROM model_metrics ORDER BY model_name"
        ).fetchall()

    assert rows == [
        ("catboost", 987.0, 750.0, 0.09, 0.41, 0.58),
        ("ridge", 1111.0, 900.0, 0.12, 0.31, 0.44),
    ]


def test_insert_metrics_is_idempotent(tmp_path):
    db_path = _db_path(tmp_path)
    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
        git_commit="abc123",
    )

    metrics = _sample_metrics()
    benchmark_db.insert_metrics(db_path, run_id, metrics)
    benchmark_db.insert_metrics(db_path, run_id, metrics)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM model_metrics WHERE run_id = ?", (run_id,)).fetchone()[0]

    assert count == 2


def test_insert_skipped_stores_reasons(tmp_path):
    db_path = _db_path(tmp_path)
    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=False,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
        git_commit="abc123",
    )

    benchmark_db.insert_skipped(db_path, run_id, {"tabpfn": "dependency missing", "autogluon": "disabled"})

    with sqlite3.connect(db_path) as conn:
        rows = conn.execute("SELECT model_name, reason FROM skipped_models ORDER BY model_name").fetchall()

    assert rows == [("autogluon", "disabled"), ("tabpfn", "dependency missing")]


def test_insert_skipped_is_idempotent(tmp_path):
    db_path = _db_path(tmp_path)
    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=False,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
        git_commit="abc123",
    )

    skipped = {"tabpfn": "dependency missing", "autogluon": "disabled"}
    benchmark_db.insert_skipped(db_path, run_id, skipped)
    benchmark_db.insert_skipped(db_path, run_id, skipped)

    with sqlite3.connect(db_path) as conn:
        count = conn.execute("SELECT COUNT(*) FROM skipped_models WHERE run_id = ?", (run_id,)).fetchone()[0]

    assert count == 2


def test_get_latest_run_returns_none_when_empty(tmp_path):
    db_path = _db_path(tmp_path)

    assert benchmark_db.get_latest_run(db_path) is None


def test_get_latest_run_returns_newest_run(tmp_path):
    db_path = _db_path(tmp_path)
    first_id = benchmark_db.insert_run(
        db_path,
        random_state=1,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir="first",
        duration_sec=1.0,
        git_commit="sha1",
    )
    second_id = benchmark_db.insert_run(
        db_path,
        random_state=2,
        train_catboost=False,
        run_tabpfn=True,
        run_tabfm=True,
        run_autogluon=True,
        autogluon_tl=900,
        output_dir="second",
        duration_sec=2.0,
        git_commit="sha2",
    )

    latest = benchmark_db.get_latest_run(db_path)

    assert first_id == 1
    assert second_id == 2
    assert latest is not None
    assert latest["id"] == 2
    assert latest["random_state"] == 2
    assert latest["output_dir"] == "second"
    assert latest["git_commit"] == "sha2"


def test_get_latest_run_includes_metrics(tmp_path):
    db_path = _db_path(tmp_path)
    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
        git_commit="abc123",
    )
    benchmark_db.insert_metrics(db_path, run_id, _sample_metrics())
    benchmark_db.insert_skipped(db_path, run_id, {"autogluon": "disabled"})

    latest = benchmark_db.get_latest_run(db_path)

    assert latest is not None
    assert latest["metrics"]["ridge"]["mae_eur"] == 1111.0
    assert latest["metrics"]["catboost"]["within_15"] == 0.58
    assert latest["skipped"] == {"autogluon": "disabled"}


def test_get_run_history_returns_dataframe(tmp_path):
    db_path = _db_path(tmp_path)
    benchmark_db.insert_run(
        db_path,
        random_state=11,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=3.5,
        git_commit="abc123",
    )

    history = benchmark_db.get_run_history(db_path)

    assert isinstance(history, pd.DataFrame)
    assert list(history.columns) == [
        "id",
        "created_at",
        "random_state",
        "train_catboost",
        "run_tabpfn",
        "run_tabfm",
        "run_autogluon",
        "autogluon_tl",
        "duration_sec",
        "model_count",
    ]
    assert history.iloc[0]["model_count"] == 0


def test_get_run_history_returns_empty_on_empty_db(tmp_path):
    db_path = _db_path(tmp_path)

    history = benchmark_db.get_run_history(db_path)

    assert isinstance(history, pd.DataFrame)
    assert history.empty
    assert list(history.columns) == [
        "id",
        "created_at",
        "random_state",
        "train_catboost",
        "run_tabpfn",
        "run_tabfm",
        "run_autogluon",
        "autogluon_tl",
        "duration_sec",
        "model_count",
    ]


def test_get_best_run_summary_returns_empty_on_empty_db(tmp_path):
    db_path = _db_path(tmp_path)

    summary = benchmark_db.get_best_run_summary(db_path)

    assert isinstance(summary, pd.DataFrame)
    assert summary.empty
    assert list(summary.columns) == [
        "model_name",
        "run_id",
        "created_at",
        "output_dir",
        "mae_eur",
        "median_ae",
        "mape",
        "within_10",
        "within_15",
        "duration_sec",
        "random_state",
        "train_catboost",
        "run_tabpfn",
        "run_tabfm",
        "run_autogluon",
        "autogluon_tl",
        "git_commit",
        "skipped_count",
        "skipped_models",
    ]


def test_get_best_run_summary_prefers_lower_mae_and_ties_by_recency_then_run_id(tmp_path):
    db_path = _db_path(tmp_path)

    older_best = benchmark_db.insert_run(
        db_path,
        random_state=1,
        train_catboost=True,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir="older",
        duration_sec=10.0,
        git_commit="sha-old",
    )
    newer_same_mae = benchmark_db.insert_run(
        db_path,
        random_state=2,
        train_catboost=False,
        run_tabpfn=True,
        run_tabfm=True,
        run_autogluon=False,
        autogluon_tl=900,
        output_dir="newer",
        duration_sec=11.0,
        git_commit="sha-new",
    )
    best_other_model = benchmark_db.insert_run(
        db_path,
        random_state=3,
        train_catboost=False,
        run_tabpfn=False,
        run_tabfm=True,
        run_autogluon=True,
        autogluon_tl=1200,
        output_dir="other",
        duration_sec=12.0,
        git_commit="sha-other",
    )
    tied_run_id_wins = benchmark_db.insert_run(
        db_path,
        random_state=4,
        train_catboost=False,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=1200,
        output_dir="other-tie",
        duration_sec=13.0,
        git_commit="sha-tie",
    )

    with sqlite3.connect(db_path) as conn:
        conn.execute("UPDATE runs SET created_at = '2024-01-01 10:00:00' WHERE id = ?", (older_best,))
        conn.execute("UPDATE runs SET created_at = '2024-01-02 10:00:00' WHERE id = ?", (newer_same_mae,))
        conn.execute("UPDATE runs SET created_at = '2024-01-02 10:00:00' WHERE id = ?", (best_other_model,))
        conn.execute("UPDATE runs SET created_at = '2024-01-02 10:00:00' WHERE id = ?", (tied_run_id_wins,))

    benchmark_db.insert_metrics(
        db_path,
        older_best,
        {
            "ridge": {
                "mae_eur": 100.0,
                "median_ae": 90.0,
                "mape": 0.10,
                "within_10": 0.50,
                "within_15": 0.60,
            },
            "catboost": {
                "mae_eur": 250.0,
                "median_ae": 200.0,
                "mape": 0.20,
                "within_10": 0.40,
                "within_15": 0.55,
            },
        },
    )
    benchmark_db.insert_metrics(
        db_path,
        newer_same_mae,
        {
            "ridge": {
                "mae_eur": 100.0,
                "median_ae": 80.0,
                "mape": 0.08,
                "within_10": 0.55,
                "within_15": 0.65,
            },
            "catboost": {
                "mae_eur": 250.0,
                "median_ae": 190.0,
                "mape": 0.19,
                "within_10": 0.42,
                "within_15": 0.57,
            },
        },
    )
    benchmark_db.insert_skipped(db_path, newer_same_mae, {"autogluon": "disabled", "tabpfn": "missing dependency"})
    benchmark_db.insert_metrics(
        db_path,
        best_other_model,
        {
            "xgboost": {
                "mae_eur": 75.0,
                "median_ae": 70.0,
                "mape": 0.07,
                "within_10": 0.65,
                "within_15": 0.72,
            }
        },
    )
    benchmark_db.insert_metrics(
        db_path,
        tied_run_id_wins,
        {
            "xgboost": {
                "mae_eur": 75.0,
                "median_ae": 68.0,
                "mape": 0.069,
                "within_10": 0.66,
                "within_15": 0.73,
            }
        },
    )

    summary = benchmark_db.get_best_run_summary(db_path)

    assert list(summary["model_name"]) == ["xgboost", "ridge", "catboost"]
    xgboost_row = summary.loc[summary["model_name"] == "xgboost"].iloc[0]
    assert xgboost_row["run_id"] == tied_run_id_wins
    assert xgboost_row["output_dir"] == "other-tie"
    ridge_row = summary.loc[summary["model_name"] == "ridge"].iloc[0]
    assert ridge_row["run_id"] == newer_same_mae
    assert ridge_row["output_dir"] == "newer"
    assert ridge_row["skipped_count"] == 2
    assert ridge_row["skipped_models"] == (("autogluon", "disabled"), ("tabpfn", "missing dependency"))
    catboost_row = summary.loc[summary["model_name"] == "catboost"].iloc[0]
    assert catboost_row["run_id"] == newer_same_mae
    assert catboost_row["git_commit"] == "sha-new"


def test_insert_run_captures_git_commit_when_available(tmp_path, monkeypatch):
    db_path = _db_path(tmp_path)
    monkeypatch.setattr(benchmark_db, "_current_git_commit", lambda: "feedface")

    run_id = benchmark_db.insert_run(
        db_path,
        random_state=42,
        train_catboost=False,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        autogluon_tl=600,
        output_dir=None,
        duration_sec=None,
    )

    latest = benchmark_db.get_latest_run(db_path)

    assert run_id == 1
    assert latest is not None
    assert latest["git_commit"] == "feedface"
