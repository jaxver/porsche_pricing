import pandas as pd

from elferspot_listings.modeling.feature_screening import screen_features


def test_screen_features_reports_selected_and_excluded_features_with_reasons():
    df = pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 130000.0, 140000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0],
            "constant": [1, 1, 1, 1],
            "mostly_null": [None, None, None, "value"],
            "Model": ["A", "B", "C", "D"],
        }
    )

    report = screen_features(
        df,
        columns=["Mileage_km", "constant", "mostly_null", "Model"],
        target_col="price_in_eur",
        max_null_fraction=0.5,
        max_cardinality=3,
        categorical_columns=("Model",),
    )

    assert report["target"] == "price_in_eur"
    assert report["target_violations"] == []
    assert report["selected_features"] == ["Mileage_km"]
    assert {item["feature"]: item["reasons"] for item in report["excluded_features"]} == {
        "constant": ["constant"],
        "mostly_null": ["null_fraction_gt_0.5", "constant", "min_frequency_lt_2"],
        "Model": ["cardinality_gt_3", "min_frequency_lt_2"],
    }


def test_screen_features_reports_target_violations():
    df = pd.DataFrame({"price_in_eur": [100000.0, None, -1.0], "Mileage_km": [1, 2, 3]})

    report = screen_features(df, columns=["Mileage_km"], target_col="price_in_eur")

    assert report["target_violations"] == ["target_has_nulls", "target_has_non_positive_values"]
