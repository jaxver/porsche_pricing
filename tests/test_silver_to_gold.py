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
    df = pd.DataFrame({"Model": ["Porsche 911 Carrera", "Cayman 982", "Something Rare"]})

    result = create_model_categories(df)

    assert result["model_category"].tolist() == ["Base Carrera / Targa / 912", "718", "Other"]


def test_create_model_categories_restores_legacy_hierarchy():
    df = pd.DataFrame(
        {
            "Model": [
                "Porsche 911 GT2 RS",
                "Porsche 911 GT3 RS",
                "Porsche 964 Carrera RS",
                "Porsche 911 Speedster",
                "Singer 911",
                "Porsche 911 Turbo S",
                "Porsche 911 GTS",
                "Porsche 911 Carrera 3.2",
                "Porsche 912 Coupe",
                "Porsche Cayman GT4",
                "Porsche Boxster",
            ]
        }
    )

    result = create_model_categories(df)

    assert result["model_category"].tolist() == [
        "GT2RS and RARE Models",
        "GT3RS",
        "RS Model",
        "Special / Backdate",
        "Bespoke / Rarest Models",
        "Turbo S / Turbo",
        "GTS",
        "Carrera 3.0/3.2 / S / SC",
        "Base Carrera / Targa / 912",
        "GT4 / GT3 / GT2",
        "718",
    ]


def test_create_model_categories_prefers_specific_match_over_generic_911():
    df = pd.DataFrame({"Model": ["Porsche 911 Carrera RS", "Porsche 911 Carrera"]})

    result = create_model_categories(df)

    assert result["model_category"].tolist() == ["RS Model", "Base Carrera / Targa / 912"]


def test_create_model_categories_preserves_legacy_edge_cases():
    df = pd.DataFrame(
        {
            "Model": [
                "Porsche 911 S",
                "Porsche 911 SC",
                "Porsche 911 RSR",
                "Porsche GT2 RSR",
                "Porsche Carrera GT",
            ]
        }
    )

    result = create_model_categories(df)

    assert result["model_category"].tolist() == [
        "Carrera 3.0/3.2 / S / SC",
        "Carrera 3.0/3.2 / S / SC",
        "GT2RS and RARE Models",
        "GT2RS and RARE Models",
        "Bespoke / Rarest Models",
    ]


def test_create_model_categories_prioritizes_718_before_generic_body_styles():
    df = pd.DataFrame({"Model": ["Porsche Boxster Cabriolet", "Porsche Cayman Coupe"]})

    result = create_model_categories(df)

    assert result["model_category"].tolist() == ["718", "718"]


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

    assert "listing_score" in result.columns
    assert result["listing_score"].dtype.kind in "fi"
    assert result.loc[0, "listing_score"] > result.loc[1, "listing_score"]
    assert result.loc[1, "listing_score"] == 0


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
