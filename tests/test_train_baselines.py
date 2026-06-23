from __future__ import annotations

import json
import importlib.util

import pandas as pd
from sklearn.model_selection import train_test_split

from elferspot_listings.modeling.baselines import MedianRegressor
from elferspot_listings.modeling.persistence import SkopsNotInstalledError
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
    skops_missing = importlib.util.find_spec("skops") is None

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
    assert result.skipped_models.get("skrub_ridge") == "skrub is not installed"
    if skops_missing:
        assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
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
    skipped_payload = json.loads((tmp_path / "skipped_models.json").read_text(encoding="utf-8"))
    assert skipped_payload.get("skrub_ridge") == "skrub is not installed"
    if skops_missing:
        assert skipped_payload.get("ridge_artifact") == "skops is not installed"


def test_train_baseline_models_clears_stale_skipped_models_file_when_skrub_recovers(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    train_baseline_models(gold_df, tmp_path, random_state=42)

    skipped_path = tmp_path / "skipped_models.json"
    skops_missing = importlib.util.find_spec("skops") is None
    assert skipped_path.exists() == skops_missing

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert result.skipped_models.get("skrub_ridge") is None
    if skops_missing:
        assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
    else:
        assert result.skipped_models.get("ridge_artifact") is None
    assert set(result.metrics) == {"median", "ridge", "skrub_ridge"}


def test_train_baseline_models_removes_stale_sklearn_artifacts_when_skops_is_unavailable(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ridge.skops").write_text("stale ridge", encoding="utf-8")
    (artifacts_dir / "skrub_ridge.skops").write_text("stale skrub", encoding="utf-8")

    def raise_skops_missing(*_args, **_kwargs):
        raise SkopsNotInstalledError("skops is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", raise_skops_missing)

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert not (artifacts_dir / "ridge.skops").exists()
    assert not (artifacts_dir / "skrub_ridge.skops").exists()
    assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
    assert result.skipped_models.get("skrub_ridge_artifact") == "skops is not installed"


def test_train_baseline_models_records_non_skops_artifact_failure_reason(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ridge.skops").write_text("stale ridge", encoding="utf-8")

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("serializer broke")

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", raise_value_error)

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert not (artifacts_dir / "ridge.skops").exists()
    assert result.skipped_models.get("ridge_artifact") == "ValueError: serializer broke"
