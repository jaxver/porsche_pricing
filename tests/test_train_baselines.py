from __future__ import annotations

import json

import pandas as pd

from elferspot_listings.modeling.baselines import MedianRegressor
from elferspot_listings.modeling.train import train_baseline_models


def _gold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0],
            "Year of construction": [1995, 1997, 2000, 2003],
            "model_category": ["911", "911", "Cayenne", "Boxster"],
        }
    )


def test_train_baseline_models_writes_reports_and_returns_metrics(tmp_path, monkeypatch):
    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=7)

    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert result.skipped_models == {"skrub_ridge": "skrub is not installed"}
    assert set(result.metrics) == {"median", "ridge"}
    assert list(result.predictions.columns) == [
        "row_index",
        "model_name",
        "actual_price_eur",
        "predicted_price_eur",
        "residual_eur",
    ]
    assert set(result.predictions["model_name"]) == {"median", "ridge"}
    assert result.predictions["model_name"].value_counts().to_dict() == {"median": 1, "ridge": 1}

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"median", "ridge"}
    assert json.loads((tmp_path / "skipped_models.json").read_text(encoding="utf-8")) == {
        "skrub_ridge": "skrub is not installed"
    }


def test_train_baseline_models_clears_stale_skipped_models_file_when_skrub_recovers(tmp_path, monkeypatch):
    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0],
            "Year of construction": [1995, 1997, 2000, 2003],
            "model_category": ["911", "911", "Cayenne", "Boxster"],
        }
    )

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    train_baseline_models(gold_df, tmp_path, random_state=7)

    skipped_path = tmp_path / "skipped_models.json"
    assert skipped_path.exists()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    result = train_baseline_models(gold_df, tmp_path, random_state=7)

    assert result.skipped_models == {}
    assert not skipped_path.exists()
    assert set(result.metrics) == {"median", "ridge", "skrub_ridge"}
