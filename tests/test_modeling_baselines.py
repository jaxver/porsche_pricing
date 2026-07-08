import sys
import types

import pytest
import pandas as pd
import numpy as np
from sklearn.utils import Tags
from sklearn.utils._tags import TargetTags

from elferspot_listings.modeling.baselines import (
    build_high_price_specialist_pipeline,
    MedianRegressor,
    build_elasticnet_pipeline,
    build_perpetual_pipeline,
    build_ridge_pipeline,
    build_skrub_ridge_pipeline,
)
from elferspot_listings.modeling.features import SelectedColumns


def _perpetual_sklearn_tags():
    return Tags(estimator_type="regressor", target_tags=TargetTags(required=False))


def test_median_regressor_predicts_global_median():
    regressor = MedianRegressor()

    regressor.fit(pd.DataFrame({"feature": [1, 2, 3]}), [10, 20, 30])

    assert regressor.predict(pd.DataFrame({"feature": [99, 100]})).tolist() == [20.0, 20.0]


def test_ridge_pipeline_fits_mixed_dataframe_and_predicts_positive_eur_scale_values():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
        text=("Description",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
            "Description": ["Sport Classic one of 30", "Standard car", None, "RS Tuning Cup S"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_ridge_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_ridge_pipeline_includes_tfidf_text_features():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
        text=("Description",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 20000, 30000, 40000],
            "Description": ["Sport Classic", "Cup S", "RS Tuning", "standard carrera"],
        }
    )

    model = build_ridge_pipeline(selected, tfidf_max_features=20, tfidf_min_df=1, tfidf_ngram_range=(1, 2))
    model.fit(X, [300000, 240000, 220000, 120000])

    text_transformer = {name: transformer for name, transformer, _columns in model.regressor_.named_steps["features"].transformers_}["text"]
    assert text_transformer.named_steps["tfidf"].max_features == 20
    assert text_transformer.named_steps["tfidf"].ngram_range == (1, 2)


def test_ridge_pipeline_excludes_redundant_linear_text_flags():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=(
            "Mileage_km",
            "model_cat_ordered",
            "Mileage_model_cat",
            "limited_production",
            "racing_history",
            "weissach_package",
            "pccb",
            "ceramic_brakes",
            "bucket_seats",
            "clubsport_package",
            "front_axle_lift",
            "sport_chrono",
            "manual_transmission_text",
            "paint_to_sample_text",
            "manthey",
            "ruf",
            "techart",
            "carbon_package",
            "lightweight_package",
            "full_leather",
            "carbon_bucket_seats",
        ),
        categorical=(),
        text=("Description",),
    )

    model = build_ridge_pipeline(selected)

    numeric_columns = next(
        columns
        for name, _transformer, columns in model.regressor.steps[0][1].transformers
        if name == "numeric"
    )
    assert numeric_columns == ["Mileage_km"]


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


def test_high_price_specialist_pipeline_fits_and_predicts_positive_values():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, 30000, 40000, 15000, 50000],
            "Year of construction": [1995, 2000, 1988, 2010, 2004, 2014],
            "model_category": ["911", "Cayenne", "Boxster", "Targa", "964", "992"],
        }
    )
    y = [120000, 150000, 260000, 340000, 310000, 420000]

    model = build_high_price_specialist_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_high_price_specialist_pipeline_routes_on_classifier_probability():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000],
            "Year of construction": [1995, 2000],
            "model_category": ["911", "Cayenne"],
        }
    )
    y = [120000, 320000]

    model = build_high_price_specialist_pipeline(selected, high_price_threshold=250000)
    model.fit(X, y)

    class FakeRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value, dtype=float)

    class FakeClassifier:
        def __init__(self, probabilities):
            self.probabilities = probabilities

        def predict_proba(self, X):
            high = np.asarray(self.probabilities, dtype=float)
            return np.column_stack([1.0 - high, high])

    model.general_regressor_ = FakeRegressor(1000.0)
    model.specialist_regressor_ = FakeRegressor(9000.0)
    model.classifier_ = FakeClassifier([0.9, 0.1])

    predictions = model.predict(X)

    assert predictions.tolist() == [9000.0, 1000.0]


def test_high_price_specialist_pipeline_falls_back_for_text_only_schema():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=(),
        categorical=(),
        text=("Description",),
    )
    X = pd.DataFrame({"Description": ["Sport Classic", "GT3 RS", "Carrera"]})

    model = build_high_price_specialist_pipeline(selected)
    model.fit(X, [100000, 200000, 300000])

    predictions = model.predict(X)

    assert predictions.tolist() == [200000.0, 200000.0, 200000.0]


def test_high_price_specialist_pipeline_uses_specialist_when_training_split_is_all_high():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )
    X = pd.DataFrame({"Mileage_km": [1000, 2000, 3000, 4000]})

    model = build_high_price_specialist_pipeline(selected, high_price_threshold=250000, min_specialist_rows=2)
    model.fit(X, [300000, 320000, 340000, 360000])

    class FakeRegressor:
        def __init__(self, value):
            self.value = value

        def predict(self, X):
            return np.full(len(X), self.value, dtype=float)

    model.general_regressor_ = FakeRegressor(1000.0)
    model.specialist_regressor_ = FakeRegressor(9000.0)
    model.classifier_ = None

    assert model.predict(X).tolist() == [9000.0] * len(X)


def test_elasticnet_pipeline_uses_ridge_preprocessing_and_predicts_positive_eur_scale_values():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km", "Year of construction"),
        categorical=("model_category",),
        text=("Description",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 25000, None, 40000],
            "Year of construction": [1995, 2000, 1988, None],
            "model_category": ["911", "Cayenne", None, "Boxster"],
            "Description": ["Sport Classic one of 30", "Standard car", None, "RS Tuning Cup S"],
        }
    )
    y = [120000, 95000, 180000, 145000]

    model = build_elasticnet_pipeline(selected)
    model.fit(X, y)
    predictions = model.predict(X)

    assert len(predictions) == len(X)
    assert (predictions > 0).all()
    assert predictions.mean() > 1000


def test_elasticnet_pipeline_includes_tfidf_text_features():
    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
        text=("Description",),
    )
    X = pd.DataFrame(
        {
            "Mileage_km": [10000, 20000, 30000, 40000],
            "Description": ["Sport Classic", "Cup S", "RS Tuning", "standard carrera"],
        }
    )

    model = build_elasticnet_pipeline(selected, tfidf_max_features=20, tfidf_min_df=1, tfidf_ngram_range=(1, 1))
    model.fit(X, [300000, 240000, 220000, 120000])

    text_transformer = {name: transformer for name, transformer, _columns in model.regressor_.named_steps["features"].transformers_}["text"]
    assert text_transformer.named_steps["tfidf"].max_features == 20
    assert text_transformer.named_steps["tfidf"].ngram_range == (1, 1)


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
            self._is_fitted = False

        def fit(self, X, y):
            captured["fit_rows"] = len(X)
            self._is_fitted = True
            return self

        def predict(self, X):
            return np.full(len(X), 123456.0)

        def __sklearn_tags__(self):
            return _perpetual_sklearn_tags()

        def __sklearn_is_fitted__(self):
            return self._is_fitted

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


def test_perpetual_pipeline_retries_without_random_state_on_value_error(monkeypatch):
    captured = {}

    class FakePerpetualRegressor:
        def __init__(self, objective, budget, random_state=None):
            if random_state is not None:
                raise ValueError("Unknown keyword arguments: dict_keys(['random_state'])")
            captured["params"] = {"objective": objective, "budget": budget}
            self._is_fitted = False

        def fit(self, X, y):
            captured["fit_rows"] = len(X)
            self._is_fitted = True
            return self

        def predict(self, X):
            return np.full(len(X), 123456.0)

        def __sklearn_tags__(self):
            return _perpetual_sklearn_tags()

        def __sklearn_is_fitted__(self):
            return self._is_fitted

    module = types.ModuleType("perpetual")
    module.PerpetualRegressor = FakePerpetualRegressor
    monkeypatch.setitem(sys.modules, "perpetual", module)

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )
    X = pd.DataFrame({"Mileage_km": [10000, 25000]})
    y = [120000, 95000]

    model = build_perpetual_pipeline(selected, random_state=17)
    model.fit(X, y)
    predictions = model.predict(X)

    assert captured["params"] == {"objective": "SquaredLoss", "budget": 0.5}
    assert captured["fit_rows"] == len(X)
    assert len(predictions) == len(X)
    assert (predictions > 0).all()


def test_perpetual_pipeline_propagates_unrelated_value_error(monkeypatch):
    class FakePerpetualRegressor:
        def __init__(self, objective, budget, random_state=None):
            raise ValueError("invalid budget")

    module = types.ModuleType("perpetual")
    module.PerpetualRegressor = FakePerpetualRegressor
    monkeypatch.setitem(sys.modules, "perpetual", module)

    selected = SelectedColumns(
        target="price_in_eur",
        numeric=("Mileage_km",),
        categorical=(),
    )

    with pytest.raises(ValueError, match="invalid budget"):
        build_perpetual_pipeline(selected, random_state=17)


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
