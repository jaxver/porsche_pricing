import numpy as np
import pandas as pd

from elferspot_listings.data_processing.silver_to_gold import (
    calculate_listing_score,
    create_log_features,
    create_model_categories,
    add_legacy_model_interaction_features,
    add_legacy_binary_flags,
    remove_outliers,
    prepare_modeling_features,
)


def test_create_log_features_adds_price_and_mileage_columns():
    df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [9999.0]})

    result = create_log_features(df)

    assert result.loc[0, "log_price"] == np.log(100000.0)
    assert result.loc[0, "log_mileage"] == np.log1p(9999.0)
    assert result.loc[0, "Mileage_sq"] == 9999.0**2


def test_remove_outliers_handles_empty_log_input():
    df = pd.DataFrame({"price_in_eur": []})

    result = remove_outliers(df, "price_in_eur", use_log=True)

    assert result.empty


def test_remove_outliers_keeps_single_positive_log_row():
    df = pd.DataFrame({"price_in_eur": [100000.0]})

    result = remove_outliers(df, "price_in_eur", use_log=True)

    assert len(result) == 1
    assert result.loc[0, "price_in_eur"] == 100000.0


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


def test_add_legacy_model_interaction_features_uses_ordered_category_code():
    df = pd.DataFrame(
        {
            "Model": ["Porsche 911 Carrera", "Porsche 911 GT2 RS"],
            "Mileage_km": [10000.0, 20000.0],
        }
    )
    df = create_log_features(df)
    df = create_model_categories(df)

    result = add_legacy_model_interaction_features(df)

    assert result["model_cat_ordered"].tolist() == [0, 8]
    assert result["inv_mileage"].tolist() == [1 / 10001.0, 1 / 20001.0]
    assert result["Mileage_model_cat"].tolist() == [0.0, 160000.0]
    assert result["inv_Mileage_model_cat"].tolist() == [0.0, 8 / 20001.0]
    assert result["Mileage_sq_model_cat"].tolist() == [0.0, (20000.0**2) * 8]


def test_add_legacy_model_interaction_features_handles_missing_mileage():
    df = pd.DataFrame({"Model": ["Unknown"], "Mileage_km": [pd.NA]})
    df = create_model_categories(df)

    result = add_legacy_model_interaction_features(df)

    assert result["model_cat_ordered"].tolist() == [11]
    assert pd.isna(result.loc[0, "inv_mileage"])
    assert pd.isna(result.loc[0, "Mileage_model_cat"])
    assert pd.isna(result.loc[0, "inv_Mileage_model_cat"])
    assert pd.isna(result.loc[0, "Mileage_sq_model_cat"])


def test_add_legacy_binary_flags_normalizes_ready_drive_and_matching_numbers():
    df = pd.DataFrame(
        {
            "Ready to drive": ["Yes", "no", ""],
            "Drive": ["Rear drive", "All wheel drive", "RWD"],
            "Matching numbers": ["Yes", "Unknown", "matching numbers"],
        }
    )

    result = add_legacy_binary_flags(df)

    assert result["state_yes"].tolist() == [1, 0, 0]
    assert result["state_Rear drive"].tolist() == [1, 0, 1]
    assert result["matching_yes"].tolist() == [1, 0, 1]


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


def test_calculate_listing_score_only_rewards_positive_matching_numbers():
    df = pd.DataFrame(
        {
            "Matching numbers": ["Yes", "matching numbers", "No", "Unknown"],
        }
    )

    result = calculate_listing_score(df)

    assert result["listing_score"].tolist() == [10, 10, 0, 0]


def test_calculate_listing_score_extracts_high_value_signals_from_all_listing_text():
    df = pd.DataFrame(
        {
            "Title": ["Porsche 992 Sport Classic bespoke commission"],
            "Model": ["997 GT3 Cup S"],
            "Description": ["One of 30 produced and prepared by RS Tuning after an engine and transmission rebuild."],
            "Secondary_Description": ["Outstanding racing history with zero running hours since overhaul."],
        }
    )

    result = calculate_listing_score(df)

    expected_flags = [
        "limited_production",
        "racing_history",
        "specialist_build",
        "bespoke_exclusive",
        "zero_running_hours",
        "engine_transmission_rebuilt",
        "cup_clubsport",
        "heritage_special",
    ]
    assert result.loc[0, expected_flags].tolist() == [1] * len(expected_flags)
    assert result.loc[0, "is_rare"] == 1
    assert result.loc[0, "is_race_ready"] == 1
    assert result.loc[0, "listing_score"] > 15


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
