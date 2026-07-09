from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pandas as pd
import pytest

from elferspot_listings.modeling.catboost_model import (
    default_catboost_params,
    fit_catboost_regressor,
    fit_catboost_quantile_interval,
    predict_catboost_eur,
    predict_catboost_interval_eur,
    save_catboost_model,
)
from elferspot_listings.modeling.features import SelectedColumns


def train_baseline_models(*args, **kwargs):
    from elferspot_listings.modeling.train import train_baseline_models as _train_baseline_models

    return _train_baseline_models(*args, **kwargs)


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
    assert "task_type" not in captured["params"]
    assert "devices" not in captured["params"]
    assert captured["pool_cat_features"] == [1, 2]
    np.testing.assert_allclose(captured["pool_label"], np.log(y.to_numpy(dtype=float)))
    assert model is not None
    assert captured["fit_pool"] is not None
    np.testing.assert_allclose(predict_catboost_eur(model, X), [210000.0, 210000.0])


def test_fit_catboost_quantile_interval_prepares_categorical_inference_features(monkeypatch):
    captured: dict[str, Any] = {"predict_frames": []}

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            captured["pool_data"] = data
            captured["pool_label"] = label
            captured["pool_cat_features"] = cat_features

    class FakeCatBoostRegressor:
        def __init__(self, **params):
            captured.setdefault("params", []).append(params)
            self.params = params

        def fit(self, pool):
            captured.setdefault("fit_pools", []).append(pool)
            return self

        def predict(self, X):
            assert X["model_category"].tolist()[0] == "Unknown"
            captured["predict_frames"].append(X.copy())
            loss_function = self.params["loss_function"]
            if loss_function == "Quantile:alpha=0.05":
                return np.log(np.full(len(X), 90000.0))
            if loss_function == "Quantile:alpha=0.5":
                return np.log(np.full(len(X), 100000.0))
            if loss_function == "Quantile:alpha=0.95":
                return np.log(np.full(len(X), 110000.0))
            raise AssertionError(loss_function)

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor, Pool=FakePool),
    )

    train = pd.DataFrame(
        {
            "price_in_eur": [80000.0, 90000.0, 120000.0, 150000.0],
            "Mileage_km": [90000.0, 70000.0, 50000.0, 30000.0],
            "model_category": ["Base Carrera / Targa / 912", "GTS", "Turbo S / Turbo", "RS Model"],
        }
    )
    selected = SelectedColumns(target="price_in_eur", numeric=("Mileage_km",), categorical=("model_category",))
    interval = fit_catboost_quantile_interval(
        train[list(selected.features)],
        train["price_in_eur"],
        selected,
        random_state=42,
        params={"iterations": 5, "depth": 2, "learning_rate": 0.1, "allow_writing_files": False, "verbose": False},
    )

    predictions = predict_catboost_interval_eur(
        interval,
        pd.DataFrame({"Mileage_km": [12345.0, 54321.0], "model_category": [None, "Turbo S / Turbo"]}),
    )

    assert interval["_selected"] == selected
    assert captured["pool_cat_features"] == [1]
    assert [params["loss_function"] for params in captured["params"]] == [
        "Quantile:alpha=0.05",
        "Quantile:alpha=0.5",
        "Quantile:alpha=0.95",
    ]
    assert list(predictions.columns) == ["pred_lower", "pred_price", "pred_upper"]
    assert len(predictions) == 2
    assert (predictions["pred_lower"] <= predictions["pred_price"]).all()
    assert (predictions["pred_price"] <= predictions["pred_upper"]).all()


def test_predict_catboost_interval_eur_handles_missing_categorical_values(monkeypatch):
    captured: dict[str, Any] = {"frames": []}

    class FakeModel:
        def __init__(self, offset: float):
            self.offset = offset

        def predict(self, X):
            captured["frames"].append(X.copy())
            assert X["model_category"].tolist()[0] == "Unknown"
            return np.log(np.full(len(X), self.offset))

    selected = SelectedColumns(target="price_in_eur", numeric=("Mileage_km",), categorical=("model_category",))
    interval = {
        "lower": FakeModel(90000.0),
        "median": FakeModel(100000.0),
        "upper": FakeModel(110000.0),
        "_selected": selected,
    }

    predictions = predict_catboost_interval_eur(
        interval,
        pd.DataFrame({"Mileage_km": [12345.0, 54321.0], "model_category": [None, "Turbo S / Turbo"]}),
    )

    assert list(predictions.columns) == ["pred_lower", "pred_price", "pred_upper"]
    assert len(predictions) == 2
    assert len(captured["frames"]) == 3
    assert (predictions["pred_lower"] <= predictions["pred_price"]).all()
    assert (predictions["pred_price"] <= predictions["pred_upper"]).all()


def test_fit_catboost_quantile_interval_returns_eur_bounds():
    pytest.importorskip("catboost")

    train = pd.DataFrame(
        {
            "price_in_eur": [80000.0, 90000.0, 120000.0, 150000.0, 250000.0, 300000.0],
            "Mileage_km": [90000.0, 70000.0, 50000.0, 30000.0, 15000.0, 10000.0],
            "model_category": [
                "Base Carrera / Targa / 912",
                "Base Carrera / Targa / 912",
                "GTS",
                "Turbo S / Turbo",
                "RS Model",
                "GT2RS and RARE Models",
            ],
        }
    )
    selected = SelectedColumns(target="price_in_eur", numeric=("Mileage_km",), categorical=("model_category",))

    interval = fit_catboost_quantile_interval(
        train[list(selected.features)],
        train["price_in_eur"],
        selected,
        random_state=42,
        params={"iterations": 5, "depth": 2, "learning_rate": 0.1, "allow_writing_files": False, "verbose": False},
    )
    predictions = predict_catboost_interval_eur(interval, train[list(selected.features)])

    assert list(predictions.columns) == ["pred_lower", "pred_price", "pred_upper"]
    assert len(predictions) == len(train)
    assert (predictions["pred_lower"] <= predictions["pred_price"]).all()
    assert (predictions["pred_price"] <= predictions["pred_upper"]).all()


def test_fit_catboost_quantile_interval_uses_quantile_alpha_losses(monkeypatch):
    captured: dict[str, Any] = {"params": []}

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            captured["pool_label"] = label
            captured["pool_cat_features"] = cat_features

    class FakeCatBoostRegressor:
        def __init__(self, **params):
            captured["params"].append(params)

        def fit(self, pool):
            captured.setdefault("fit_pools", []).append(pool)
            return self

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor, Pool=FakePool),
    )

    train = pd.DataFrame(
        {
            "price_in_eur": [80000.0, 90000.0, 120000.0],
            "Mileage_km": [90000.0, 70000.0, 50000.0],
            "model_category": ["Base Carrera / Targa / 912", "GTS", "Turbo S / Turbo"],
        }
    )
    selected = SelectedColumns(target="price_in_eur", numeric=("Mileage_km",), categorical=("model_category",))

    fit_catboost_quantile_interval(train[list(selected.features)], train["price_in_eur"], selected)

    assert [params["loss_function"] for params in captured["params"]] == [
        "Quantile:alpha=0.05",
        "Quantile:alpha=0.5",
        "Quantile:alpha=0.95",
    ]
    assert captured["pool_cat_features"] == [1]


def test_fit_catboost_regressor_merges_gpu_params_with_tuned_params(monkeypatch):
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

    monkeypatch.setitem(
        sys.modules,
        "catboost",
        SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor, Pool=FakePool),
    )

    X = pd.DataFrame(
        {
            "Mileage_km": [10000.0, 20000.0],
            "model_category": ["911", "Cayenne"],
        }
    )
    y = pd.Series([100000.0, 120000.0])
    selected = SelectedColumns(target="price_in_eur", numeric=("Mileage_km",), categorical=("model_category",))

    fit_catboost_regressor(
        X,
        y,
        selected,
        random_state=99,
        params={"iterations": 321, "depth": 5},
        device="gpu",
        gpu_devices="0",
    )

    assert captured["params"]["iterations"] == 321
    assert captured["params"]["depth"] == 5
    assert captured["params"]["task_type"] == "GPU"
    assert captured["params"]["devices"] == "0"


def test_train_baseline_models_tunes_catboost_on_gpu_and_merges_params(tmp_path, monkeypatch):
    from elferspot_listings.modeling.baselines import MedianRegressor

    captured: dict[str, Any] = {}

    class FakePool:
        def __init__(self, data, label=None, cat_features=None):
            captured.setdefault("pool_calls", []).append((data.copy(), None if label is None else label.copy(), list(cat_features or [])))

    class FakeCatBoostRegressor:
        def __init__(self, **params):
            captured["params"] = params

        def fit(self, pool):
            captured["fit_pool"] = pool
            return self

        def predict(self, X):
            return np.log(np.full(len(X), 210000.0))

        def save_model(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake catboost artifact\n", encoding="utf-8")

    fake_catboost = SimpleNamespace(CatBoostRegressor=FakeCatBoostRegressor, Pool=FakePool)
    monkeypatch.setitem(sys.modules, "catboost", fake_catboost)
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.tune_catboost_params",
        lambda *args, **kwargs: {"iterations": 123, "learning_rate": 0.03, "depth": 4},
    )

    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=17,
        train_catboost=True,
        tune_catboost=True,
        models=["catboost"],
        device="gpu",
        gpu_devices="0",
    )

    assert captured["params"]["iterations"] == 123
    assert captured["params"]["learning_rate"] == 0.03
    assert captured["params"]["depth"] == 4
    assert captured["params"]["task_type"] == "GPU"
    assert captured["params"]["devices"] == "0"
    assert "catboost" in result.metrics


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
    assert set(result.metrics) == {"median", "ridge", "elasticnet", "stacked_ensemble"}
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

    def fake_fit_catboost_regressor(X_train, y_train, selected, random_state=42, params=None, device="cpu", gpu_devices=None):
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


def test_train_baseline_models_can_tune_catboost_with_optuna(tmp_path, monkeypatch):
    from elferspot_listings.modeling.baselines import MedianRegressor

    captured: dict[str, object] = {}

    class FakeCatBoostModel:
        def predict(self, X):
            return np.log(np.full(len(X), 210000.0))

        def save_model(self, path):
            Path(path).parent.mkdir(parents=True, exist_ok=True)
            Path(path).write_text("fake catboost artifact\n", encoding="utf-8")

    def fake_tune_catboost_params(X_train, y_train, selected, random_state=42, n_trials=25, device="cpu", gpu_devices=None):
        captured["n_trials"] = n_trials
        captured["random_state"] = random_state
        captured["train_rows"] = len(X_train)
        return {"iterations": 123, "learning_rate": 0.03, "depth": 4}

    def fake_fit_catboost_regressor(X_train, y_train, selected, random_state=42, params=None, device="cpu", gpu_devices=None):
        captured["fit_params"] = params
        return FakeCatBoostModel()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.tune_catboost_params", fake_tune_catboost_params)
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", fake_fit_catboost_regressor)

    gold_df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=17,
        train_catboost=True,
        tune_catboost=True,
        tuning_trials=9,
    )

    assert captured["n_trials"] == 9
    assert captured["random_state"] == 17
    assert captured["train_rows"] == 6
    assert captured["fit_params"] == {"iterations": 123, "learning_rate": 0.03, "depth": 4}
    assert "catboost" in result.metrics


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

    assert "catboost" not in result.skipped_models
    assert not stale_artifact.exists()

    stale_artifact.write_text("stale artifact\n", encoding="utf-8")

    def raise_import_error(*_args, **_kwargs):
        raise ImportError("catboost is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", raise_import_error)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, train_catboost=True)

    assert result.skipped_models["catboost"] == "catboost is not installed"
    assert not stale_artifact.exists()
