"""Benchmark report writing helpers."""

from __future__ import annotations

import json
from pathlib import Path


def write_benchmark_report(metrics: dict, output_dir: str | Path) -> Path:
    """Write benchmark metrics to `metrics.json` in the output directory."""
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    report_path = output_path / "metrics.json"
    with report_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, indent=2, sort_keys=True)
        handle.write("\n")

    return report_path
