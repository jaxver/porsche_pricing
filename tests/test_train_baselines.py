from __future__ import annotations

import json
import importlib.util
import tempfile
from pathlib import Path

import pandas as pd

from elferspot_listings.modeling.train import train_baseline_models


def test_train_baseline_models_writes_reports_and_returns_metrics():
    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0],
            "Year of construction": [1995, 1997, 2000, 2003],
            "model_category": ["911", "911", "Cayenne", "Boxster"],
        }
    )

    with tempfile.TemporaryDirectory(dir=Path(r"C:\Users\USER\AppData\Local\Temp\opencode")) as temp_dir:
        output_dir = Path(temp_dir)
        result = train_baseline_models(gold_df, output_dir, random_state=7)

        metrics_path = output_dir / "metrics.json"
        predictions_path = output_dir / "predictions.csv"

        assert metrics_path.exists()
        assert predictions_path.exists()
        assert "median" in result.metrics
        assert "ridge" in result.metrics

        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        assert "median" in metrics
        assert "ridge" in metrics

        if importlib.util.find_spec("skrub") is None:
            assert result.skipped_models == {"skrub_ridge": "skrub is not installed"}
            skipped_path = output_dir / "skipped_models.json"
            assert skipped_path.exists()
            assert json.loads(skipped_path.read_text(encoding="utf-8")) == {
                "skrub_ridge": "skrub is not installed"
            }
            assert "skrub_ridge" not in metrics
        else:
            assert "skrub_ridge" in result.metrics
