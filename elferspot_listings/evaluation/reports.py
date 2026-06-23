"""Benchmark report writing helpers."""

from __future__ import annotations

import json
import math
from pathlib import Path


def _sanitize_for_json(value):
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, dict):
        return {key: _sanitize_for_json(item) for key, item in value.items()}
    if isinstance(value, list):
        return [_sanitize_for_json(item) for item in value]
    if isinstance(value, tuple):
        return [_sanitize_for_json(item) for item in value]
    return value


def write_benchmark_report(metrics: dict, output_dir: str | Path) -> Path:
    """Write benchmark metrics to `metrics.json` in the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "metrics.json"
    payload = _sanitize_for_json(metrics)
    with report_path.open("w", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False))
        handle.write("\n")

    return report_path
