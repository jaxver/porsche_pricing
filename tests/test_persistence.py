from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
import pytest
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from elferspot_listings.modeling.persistence import (
    inspect_sklearn_model,
    save_sklearn_model,
    write_model_card,
)
from elferspot_listings.modeling.train import train_baseline_models


def _tiny_regression_frame() -> tuple[pd.DataFrame, pd.Series]:
    X = pd.DataFrame({"feature": [0.0, 1.0, 2.0, 3.0]})
    y = pd.Series([1.0, 2.0, 3.0, 4.0])
    return X, y


def _gold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )


def test_write_model_card_contains_core_metadata(tmp_path):
    card_path = write_model_card(
        tmp_path / "MODEL_CARD.md",
        {
            "model_name": "ridge",
            "purpose": "Predict Porsche listing prices",
            "target": "price_in_eur",
            "metrics": {"mae_eur": 1234.5},
            "limitations": ["Synthetic test data only"],
            "usage_notes": ["Load with trusted types from skops.io"],
        },
    )

    content = card_path.read_text(encoding="utf-8")
    assert card_path.name == "MODEL_CARD.md"
    assert "# Model Card: ridge" in content
    assert "Predict Porsche listing prices" in content
    assert "## Target" in content
    assert "`price_in_eur`" in content
    assert "mae_eur" in content
    assert "Synthetic test data only" in content
    assert "Load with trusted types from skops.io" in content


def test_save_and_inspect_sklearn_model_roundtrip(tmp_path):
    skops_io = pytest.importorskip("skops.io")

    X, y = _tiny_regression_frame()
    model = Pipeline([("scale", StandardScaler()), ("ridge", Ridge())])
    model.fit(X, y)

    model_path = save_sklearn_model(model, tmp_path / "ridge.skops")
    trusted = inspect_sklearn_model(model_path)

    assert model_path.exists()
    assert isinstance(trusted, list)
    assert all(isinstance(item, str) for item in trusted)

    loaded = skops_io.load(model_path, trusted=trusted)
    assert loaded.predict(X).tolist() == pytest.approx(model.predict(X).tolist())


def test_train_baseline_models_writes_model_card_and_optional_artifacts(tmp_path, monkeypatch):
    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=42)

    model_card_path = tmp_path / "MODEL_CARD.md"
    ridge_artifact_path = tmp_path / "artifacts" / "ridge.skops"

    assert model_card_path.exists()
    assert result.skipped_models.get("skrub_ridge") == "skrub is not installed"
    if importlib.util.find_spec("skops") is None:
        assert result.skipped_models.get("ridge_artifact") == "skops is not installed"

    if importlib.util.find_spec("skops") is not None:
        assert ridge_artifact_path.exists()
    else:
        assert not ridge_artifact_path.exists()
