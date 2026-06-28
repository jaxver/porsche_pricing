import sys
import types

import pytest
import pandas as pd
import numpy as np

from elferspot_listings.modeling.baselines import (
    MedianRegressor,
    build_elasticnet_pipeline,
    build_perpetual_pipeline,
    build_ridge_pipeline,
    build_skrub_ridge_pipeline,
)
from elferspot_listings.modeling.features import SelectedColumns


def test_median_regressor_predicts_global_median():
    regressor = MedianRegressor()

    regressor.fit(pd.DataFrame({"feature": [1, 2, 3]}), [10, 20, 30])

    assert regressor.predict(pd.DataFrame({"feature": [99, 100]})).tolist() == [20.0, 20.0]


def test_ridge_pipeline_fits_mixed_dataframe_and_predicts_positive_eur_scale_values():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_ridge_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_ridge_pipeline_fits_numeric_only_schema():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=(),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_ridge_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()


def test_ridge_pipeline_fits_categorical_only_schema():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=(),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "model_category": ["911", "Cayenne", None, "Boxster"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_ridge_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()


@pytest.mark.parametrize("target_values", ([0, 100, 200], [-10, 100, 200]))
def test_ridge_pipeline_rejects_non_positive_targets(target_values):
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )
    X = pd.DataFrame({"Mileage_km": [10000, 25000, 40000]})

    model = build_ridge_pipeline(selected)

    with pytest.raises(ValueError, match="positive"):
        model.fit(X, target_values)


def test_elasticnet_pipeline_uses_ridge_preprocessing_and_predicts_positive_eur_scale_values():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_elasticnet_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_skrub_ridge_pipeline_fits_mixed_dataframe_and_predicts_positive_eur_scale_values():
    pytest.importorskip("skrub")

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_skrub_ridge_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_skrub_ridge_pipeline_rejects_non_positive_targets():
    pytest.importorskip("skrub")

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )
    X = pd.DataFrame({"Mileage_km": [10000, 25000, 40000]})

    model = build_skrub_ridge_pipeline(selected)

    with pytest.raises(ValueError, match="positive"):
        model.fit(X, [100000, 0, 200000])


def test_xgboost_pipeline_uses_cuda_device_when_requested(monkeypatch):
    from elferspot_listings.modeling.baselines import build_xgboost_pipeline

    captured = {}

    class FakeXGBRegressor:
        def __init__(self, **params):
            captured["params"] = params

    module = types.ModuleType("xgboost")
    module.XGBRegressor = FakeXGBRegressor
    monkeypatch.setitem(sys.modules, "xgboost", module)

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )

    build_xgboost_pipeline(selected, device="gpu")

    assert captured["params"]["tree_method"] == "hist"
    assert captured["params"]["device"] == "cuda"


def test_perpetual_pipeline_fits_mixed_dataframe_and_falls_back_when_random_state_is_unsupported(monkeypatch):
    captured = {}

    class FakePerpetualRegressor:
        def __init__(self, objective, budget):
            captured["params"] = {"objective": objective, "budget": budget}

        def fit(self, X, y):
            captured["fit_rows"] = len(X)
            return self

        def predict(self, X):
            return np.full(len(X), 123456.0)

    module = types.ModuleType("perpetual")
    module.PerpetualRegressor = FakePerpetualRegressor
    monkeypatch.setitem(sys.modules, "perpetual", module)

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_perpetual_pipeline(selected, random_state=17)
    model.fit(X, y)
    predictions = model.predict(X)

    assert captured["params"] == {"objective": "SquaredLoss", "budget": 0.5}
    assert captured["fit_rows"] == len(X)
    assert len(predictions) == len(X)
    assert (predictions > 0).all()


def test_perpetual_pipeline_raises_when_dependency_is_missing(monkeypatch):
    monkeypatch.delitem(sys.modules, "perpetual", raising=False)
    monkeypatch.setitem(sys.modules, "perpetual", None)

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )

    with pytest.raises(ImportError, match="perpetual is not installed"):
        build_perpetual_pipeline(selected)
