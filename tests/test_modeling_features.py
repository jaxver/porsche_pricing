import pandas as pd
import pytest

from elferspot_listings.modeling.features import (
    CATEGORICAL_ALLOWLIST,
    NUMERIC_ALLOWLIST,
    TARGET_COLUMN,
    TEXT_ALLOWLIST,
    build_feature_frame,
    select_model_columns,
)


def test_select_model_columns_returns_existing_allowlisted_columns_only():
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [100000, 120000],
            "Mileage_km": [10000, 20000],
            "Model": ["911", "Cayenne"],
            "Description": ["Sport Classic one of 30", "Standard car"],
            "unused": [1, 2],
        }
    )

    selected = select_model_columns(df)

    assert selected.target == TARGET_COLUMN
    assert selected.numeric == ("Mileage_km",)
    assert selected.categorical == ("Model",)
    assert selected.text == ("Description",)
    assert selected.features == ["Mileage_km", "Model", "Description"]


def test_select_model_columns_includes_richer_existing_gold_columns():
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [100000],
            "Mileage_km": [10000],
            "log_mileage": [9.21],
            "Mileage_sq": [100000000],
            "Transmission": ["Manual"],
            "Drive": ["RWD"],
            "Ready to drive": ["Yes"],
            "Car location": ["Germany"],
            "price_inflation_factor": [1.0],
        }
    )

    selected = select_model_columns(df)

    assert selected.numeric == ("Mileage_km", "log_mileage", "Mileage_sq", "price_inflation_factor")
    assert selected.categorical == (
        "Transmission",
        "Drive",
        "Ready to drive",
        "Car location",
    )


def test_select_model_columns_includes_restored_legacy_numeric_features():
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [100000],
            "model_cat_ordered": [1],
            "inv_mileage": [0.0001],
            "Mileage_model_cat": [10000.0],
            "inv_Mileage_model_cat": [0.0001],
            "Mileage_sq_model_cat": [100000000.0],
            "restoration_full": [1],
            "restoration_partial": [0],
            "is_restomod": [0],
            "has_docs": [1],
            "is_matching_numbers": [1],
            "is_mint": [0],
            "is_race_ready": [0],
            "is_rare": [1],
            "is_accident_free": [1],
            "has_upgrades": [0],
            "first_owner": [1],
            "state_yes": [1],
            "state_Rear drive": [1],
            "matching_yes": [1],
            "limited_production": [1],
            "racing_history": [1],
            "specialist_build": [1],
            "bespoke_exclusive": [1],
            "zero_running_hours": [1],
            "engine_transmission_rebuilt": [1],
            "cup_clubsport": [1],
            "heritage_special": [1],
            "weissach_package": [1],
            "pccb": [1],
            "ceramic_brakes": [1],
            "bucket_seats": [1],
            "clubsport_package": [1],
            "front_axle_lift": [1],
            "sport_chrono": [1],
            "manual_transmission_text": [1],
            "paint_to_sample_text": [1],
            "manthey": [1],
            "ruf": [1],
            "techart": [1],
            "carbon_package": [1],
            "lightweight_package": [1],
            "full_leather": [1],
            "carbon_bucket_seats": [1],
        }
    )

    selected = select_model_columns(df)

    assert selected.numeric == (
        "model_cat_ordered",
        "inv_mileage",
        "Mileage_model_cat",
        "inv_Mileage_model_cat",
        "Mileage_sq_model_cat",
        "restoration_full",
        "restoration_partial",
        "is_restomod",
        "has_docs",
        "is_matching_numbers",
        "is_mint",
        "is_race_ready",
        "is_rare",
        "is_accident_free",
        "has_upgrades",
        "first_owner",
        "state_yes",
        "state_Rear drive",
        "matching_yes",
        "limited_production",
        "racing_history",
        "specialist_build",
        "bespoke_exclusive",
        "zero_running_hours",
        "engine_transmission_rebuilt",
        "cup_clubsport",
        "heritage_special",
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
    )


def test_selected_columns_fields_are_immutable_tuples():
    selected = select_model_columns(
        pd.DataFrame(
            {
                TARGET_COLUMN: [100000],
                "Mileage_km": [10000],
                "Model": ["911"],
            }
        )
    )

    assert isinstance(selected.numeric, tuple)
    assert isinstance(selected.categorical, tuple)
    assert isinstance(selected.text, tuple)
    with pytest.raises(AttributeError):
        selected.numeric.append("extra")


def test_allowlists_are_immutable_tuples():
    assert isinstance(NUMERIC_ALLOWLIST, tuple)
    assert isinstance(CATEGORICAL_ALLOWLIST, tuple)
    assert isinstance(TEXT_ALLOWLIST, tuple)


def test_build_feature_frame_drops_rows_without_target():
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [100000, None, 120000],
            "Mileage_km": [10000, 20000, 30000],
            "Model": ["911", "Cayenne", "Boxster"],
        }
    )

    X, y, selected = build_feature_frame(df)

    assert len(X) == 2
    assert y.tolist() == [100000.0, 120000.0]
    assert y.dtype.kind == "f"
    assert selected.features == ["Mileage_km", "Model", "listing_text"]


def test_build_feature_frame_raises_for_missing_target():
    df = pd.DataFrame({"Mileage_km": [10000]})

    with pytest.raises(ValueError, match="target"):
        build_feature_frame(df)


def test_build_feature_frame_raises_for_no_supported_features():
    df = pd.DataFrame({TARGET_COLUMN: [100000]})

    with pytest.raises(ValueError, match="supported features"):
        build_feature_frame(df)
