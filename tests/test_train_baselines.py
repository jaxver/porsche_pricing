from __future__ import annotations

import json

import pandas as pd
from sklearn.model_selection import train_test_split

from elferspot_listings.modeling.baselines import MedianRegressor
from elferspot_listings.modeling.train import train_baseline_models


def _gold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )


def _expected_holdout_indices(frame: pd.DataFrame) -> list[int]:
    train_index, test_index = train_test_split(frame.index, test_size=0.25, random_state=42)
    return list(test_index)


def test_train_baseline_models_writes_reports_and_returns_metrics(tmp_path, monkeypatch):
    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)

    gold_df = _gold_frame()
    result = train_baseline_models(gold_df, tmp_path, random_state=42)
    expected_holdout = _expected_holdout_indices(gold_df)

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
    assert result.predictions["model_name"].value_counts().to_dict() == {"median": 2, "ridge": 2}
    for model_name in ("median", "ridge"):
        model_rows = result.predictions[result.predictions["model_name"] == model_name]
        assert model_rows["row_index"].tolist() == expected_holdout
        assert len(model_rows) == len(expected_holdout)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"median", "ridge"}
    assert json.loads((tmp_path / "skipped_models.json").read_text(encoding="utf-8")) == {
        "skrub_ridge": "skrub is not installed"
    }


def test_train_baseline_models_clears_stale_skipped_models_file_when_skrub_recovers(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    train_baseline_models(gold_df, tmp_path, random_state=42)

    skipped_path = tmp_path / "skipped_models.json"
    assert skipped_path.exists()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert result.skipped_models == {}
    assert not skipped_path.exists()
    assert set(result.metrics) == {"median", "ridge", "skrub_ridge"}
