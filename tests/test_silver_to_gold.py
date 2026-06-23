import numpy as np
import pandas as pd

from elferspot_listings.data_processing.silver_to_gold import (
    calculate_listing_score,
    create_log_features,
    create_model_categories,
    prepare_modeling_features,
)


def test_create_log_features_adds_price_and_mileage_columns():
    df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [9999.0]})

    result = create_log_features(df)

    assert result.loc[0, "log_price"] == np.log(100000.0)
    assert result.loc[0, "log_mileage"] == np.log1p(9999.0)
    assert result.loc[0, "Mileage_sq"] == 9999.0**2


def test_create_model_categories_maps_known_models():
    df = pd.DataFrame({"Model": ["Porsche 911 Carrera", "Cayenne S", "Something Rare"]})

    result = create_model_categories(df)

    assert result["model_category"].tolist() == ["911", "SUV", "Other"]


def test_calculate_listing_score_uses_available_quality_fields():
    df = pd.DataFrame(
        {
            "Matching numbers": ["Yes", "Unknown"],
            "Number of vehicle owners": ["2", "Unknown"],
            "Interior color": ["Black", None],
            "Exterior color": ["Silver", None],
            "Paint-to-Sample (PTS)": [1, 0],
            "is_fully_restored": [1, 0],
            "Mileage_km": [30000, 80000],
        }
    )

    result = calculate_listing_score(df)

    assert result["listing_score"].tolist() == [75, 0]


def test_prepare_modeling_features_coerces_numeric_and_fills_colors():
    df = pd.DataFrame(
        {
            "Mileage_km": ["10000"],
            "price_in_eur": ["120000"],
            "Year of construction": ["1996"],
            "Interior color": [None],
            "Exterior color": [None],
        }
    )

    result = prepare_modeling_features(df)

    assert result["Mileage_km"].dtype.kind in "fi"
    assert result.loc[0, "Interior color"] == "Unknown"
    assert result.loc[0, "Exterior color"] == "Unknown"
