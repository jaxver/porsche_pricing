import json
import tempfile
from pathlib import Path

import pandas as pd

from elferspot_listings.evaluation.metrics import regression_metrics, segment_metrics
from elferspot_listings.evaluation.reports import write_benchmark_report


def test_regression_metrics_computes_error_summary():
    result = regression_metrics([100.0, 200.0], [110.0, 180.0])

    assert result["mae_eur"] == 15.0
    assert result["median_ae_eur"] == 15.0
    assert result["mape"] == 0.1
    assert result["within_10pct"] == 1.0
    assert result["within_15pct"] == 1.0


def test_segment_metrics_groups_by_available_columns_and_skips_missing_columns():
    df = pd.DataFrame(
        {
            "actual": [100.0, 120.0, 140.0, 160.0],
            "predicted": [110.0, 100.0, 150.0, 170.0],
            "region": ["A", "A", "B", "B"],
            "fuel": ["Gas", "Electric", "Gas", "Electric"],
        }
    )

    result = segment_metrics(
        df,
        actual_col="actual",
        predicted_col="predicted",
        segment_cols=["region", "missing", "fuel"],
    )

    assert list(result["segment_column"]) == ["region", "region", "fuel", "fuel"]
    assert list(result["segment_value"]) == ["A", "B", "Electric", "Gas"]
    assert list(result["n_rows"]) == [2, 2, 2, 2]
    assert "missing" not in result["segment_column"].tolist()
    assert set(result.columns) >= {
        "segment_column",
        "segment_value",
        "n_rows",
        "mae_eur",
        "median_ae_eur",
        "mape",
        "within_10pct",
        "within_15pct",
    }


def test_write_benchmark_report_writes_metrics_json():
    with tempfile.TemporaryDirectory() as temp_dir:
        output_path = write_benchmark_report({"mae_eur": 15.0, "mape": 0.1}, Path(temp_dir))

        assert output_path.name == "metrics.json"
        assert output_path.exists()
        with output_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
        assert data == {"mae_eur": 15.0, "mape": 0.1}
