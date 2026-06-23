from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path

import pandas as pd

from elferspot_listings.utils.dashboard_data import (
    find_latest_benchmark_run,
    load_latest_metrics,
    load_latest_predictions,
)


TEMP_ROOT = Path(r"C:\Users\USER\AppData\Local\Temp\opencode")


def _write_run(
    run_dir: Path,
    *,
    prediction_rows: list[dict] | None,
    metrics: dict | None,
    timestamp: float,
) -> Path:
    run_dir.mkdir(parents=True, exist_ok=True)

    if prediction_rows is not None:
        predictions_path = run_dir / "predictions.csv"
        pd.DataFrame(prediction_rows).to_csv(predictions_path, index=False)
        os.utime(predictions_path, (timestamp, timestamp))

    if metrics is not None:
        metrics_path = run_dir / "metrics.json"
        metrics_path.write_text(json.dumps(metrics), encoding="utf-8")
        os.utime(metrics_path, (timestamp, timestamp))

    os.utime(run_dir, (timestamp, timestamp))
    return run_dir


def test_find_latest_benchmark_run_uses_predictions_mtime_and_keeps_metrics_in_same_run():
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp_dir:
        results_dir = Path(temp_dir) / "results" / "benchmarks"

        _write_run(
            results_dir / "2026-01-01",
            prediction_rows=[{"row_index": 1, "model_name": "ridge", "predicted_price_eur": 100.0}],
            metrics={"ridge": {"mae_eur": 10.0}},
            timestamp=1000.0,
        )
        _write_run(
            results_dir / "2026-01-02",
            prediction_rows=None,
            metrics={"ridge": {"mae_eur": 1.0}},
            timestamp=3000.0,
        )
        latest_run = _write_run(
            results_dir / "2026-01-03",
            prediction_rows=[
                {"row_index": 2, "model_name": "ridge", "predicted_price_eur": 200.0},
                {"row_index": 3, "model_name": "catboost", "predicted_price_eur": 250.0},
            ],
            metrics={"ridge": {"mae_eur": 8.0}, "catboost": {"mae_eur": 6.0}},
            timestamp=2000.0,
        )

        assert find_latest_benchmark_run(results_dir) == latest_run

        predictions = load_latest_predictions(results_dir)
        metrics = load_latest_metrics(results_dir)

        assert predictions is not None
        assert list(predictions["predicted_price_eur"]) == [200.0, 250.0]
        assert metrics == {"ridge": {"mae_eur": 8.0}, "catboost": {"mae_eur": 6.0}}


def test_dashboard_helpers_return_none_when_predictions_are_missing():
    with tempfile.TemporaryDirectory(dir=TEMP_ROOT) as temp_dir:
        results_dir = Path(temp_dir) / "results" / "benchmarks"
        metrics_only_run = results_dir / "2026-01-05"
        metrics_only_run.mkdir(parents=True, exist_ok=True)
        (metrics_only_run / "metrics.json").write_text(json.dumps({"ridge": {"mae_eur": 1.0}}), encoding="utf-8")

        assert find_latest_benchmark_run(results_dir) is None
        assert load_latest_predictions(results_dir) is None
        assert load_latest_metrics(results_dir) is None
