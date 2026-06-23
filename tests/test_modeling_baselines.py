import pytest
import pandas as pd

from elferspot_listings.modeling.baselines import (
    MedianRegressor,
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
