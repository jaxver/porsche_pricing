"""Helpers for loading the latest benchmark dashboard artifacts."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd


def find_latest_benchmark_run(results_dir: str | Path = "results/benchmarks") -> Path | None:
    """Return the run directory with the newest `predictions.csv`.

    If no predictions exist, return `None` rather than guessing from metrics-only runs.
    """

    base_dir = Path(results_dir)
    if not base_dir.exists():
        return None

    latest_run: Path | None = None
    latest_mtime: float | None = None

    for predictions_path in base_dir.glob("*/predictions.csv"):
        try:
            mtime = predictions_path.stat().st_mtime
        except OSError:
            continue

        if latest_mtime is None or mtime > latest_mtime:
            latest_mtime = mtime
            latest_run = predictions_path.parent

    return latest_run


def load_latest_predictions(results_dir: str | Path = "results/benchmarks") -> pd.DataFrame | None:
    """Load predictions from the latest benchmark run, if available."""

    benchmark_run = find_latest_benchmark_run(results_dir)
    if benchmark_run is None:
        return None

    predictions_path = benchmark_run / "predictions.csv"
    try:
        return pd.read_csv(predictions_path)
    except (OSError, pd.errors.EmptyDataError, ValueError, UnicodeDecodeError):
        return None


def load_latest_metrics(results_dir: str | Path = "results/benchmarks") -> dict | None:
    """Load metrics from the same benchmark run as the latest predictions."""

    benchmark_run = find_latest_benchmark_run(results_dir)
    if benchmark_run is None:
        return None

    metrics_path = benchmark_run / "metrics.json"
    try:
        return json.loads(metrics_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError, UnicodeDecodeError):
        return None
