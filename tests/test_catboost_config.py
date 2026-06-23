from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd

from elferspot_listings.modeling.catboost_model import (
    default_catboost_params,
    fit_catboost_regressor,
    predict_catboost_eur,
    save_catboost_model,
)
from elferspot_listings.modeling.features import SelectedColumns
from elferspot_listings.modeling.train import train_baseline_models


def test_default_catboost_params_returns_reproducible_quiet_configuration():
    params_a = default_catboost_params(random_state=123)
    params_b = default_catboost_params(random_state=123)

    assert params_a == params_b
    assert params_a["loss_function"] == "RMSE"
    assert params_a["random_seed"] == 123
    assert params_a["verbose"] is False
    assert params_a["allow_writing_files"] is False


def test_fit_catboost_regressor_uses_log_target_and_selected_categorical_columns(monkeypatch):
    captured: dict[str, Any] = {}

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            captured["pool_data"] = data
            captured["pool_label"] = label
            captured["pool_cat_features"] = cat_features

    class FakeCatBoostRegressor:
        def __init__(self, **params):
            captured["params"] = params

        def fit(self, pool):
            captured["fit_pool"] = pool
            return self

        def predict(self, X):
            return np.log(np.full(len(X), 210000.0))

        def save_model(self, path):
            Path(path).write_text("fake catboost artifact\n", encoding="utf-8")

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor, Pool=FakePool),
    )

    X = pd.DataFrame(
        {
            "Mileage_km": [10000.0, 20000.0],
            "model_category": ["911", "Cayenne"],
            "Interior color": ["Black", "Tan"],
        }
    )
    y = pd.Series([100000.0, 120000.0])
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=("missing", "model_category", "Interior color"),
    )

    model = fit_catboost_regressor(X, y, selected, random_state=99)

    assert captured["params"]["random_seed"] == 99
    assert captured["params"]["loss_function"] == "RMSE"
    assert captured["pool_cat_features"] == [1, 2]
    np.testing.assert_allclose(captured["pool_label"], np.log(y.to_numpy(dtype=float)))
    assert model is not None
    assert captured["fit_pool"] is not None
    np.testing.assert_allclose(predict_catboost_eur(model, X), [210000.0, 210000.0])


def test_save_catboost_model_creates_parent_dir_and_saves_native_artifact(tmp_path):
    class FakeModel:
        def __init__(self):
            self.saved_path = None

        def save_model(self, path):
            self.saved_path = Path(path)
            self.saved_path.write_text("saved\n", encoding="utf-8")

    model = FakeModel()
    output_path = tmp_path / "artifacts" / "catboost.cbm"

    save_catboost_model(model, output_path)

    assert output_path.exists()
    assert model.saved_path == output_path


def test_train_baseline_models_skips_catboost_when_unavailable(tmp_path, monkeypatch):
    from elferspot_listings.modeling.baselines import MedianRegressor

    def raise_import_error(*_args, **_kwargs):
        raise ImportError("catboost is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", raise_import_error)

    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, train_catboost=True)

    assert result.skipped_models["catboost"] == "catboost is not installed"
    assert set(result.metrics) == {"median", "ridge"}
    assert not (tmp_path / "artifacts" / "catboost.cbm").exists()


def test_train_baseline_models_includes_catboost_when_enabled(tmp_path, monkeypatch):
    from elferspot_listings.modeling.baselines import MedianRegressor

    class FakeCatBoostModel:
        def __init__(self):
            self.saved_path = None

        def predict(self, X):
            return np.log(np.full(len(X), 210000.0))

        def save_model(self, path):
            self.saved_path = Path(path)
            self.saved_path.write_text("fake catboost artifact\n", encoding="utf-8")

    def fake_fit_catboost_regressor(X_train, y_train, selected, random_state=42):
        return FakeCatBoostModel()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", fake_fit_catboost_regressor)

    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, train_catboost=True)

    assert "catboost" in result.metrics
    assert "catboost" in set(result.predictions["model_name"])
    assert (tmp_path / "artifacts" / "catboost.cbm").exists()


def test_train_baseline_models_removes_stale_catboost_artifact_when_not_training(tmp_path, monkeypatch):
    from elferspot_listings.modeling.baselines import MedianRegressor

    stale_artifact = tmp_path / "artifacts" / "catboost.cbm"
    stale_artifact.parent.mkdir(parents=True, exist_ok=True)
    stale_artifact.write_text("stale artifact\n", encoding="utf-8")

    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())

    result = train_baseline_models(gold_df, tmp_path, random_state=42, train_catboost=False)

    assert result.skipped_models == {}
    assert not stale_artifact.exists()

    stale_artifact.write_text("stale artifact\n", encoding="utf-8")

    def raise_import_error(*_args, **_kwargs):
        raise ImportError("catboost is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", raise_import_error)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, train_catboost=True)

    assert result.skipped_models["catboost"] == "catboost is not installed"
    assert not stale_artifact.exists()
