import pandas as pd
import pytest

from elferspot_listings.modeling.features import (
    CATEGORICAL_ALLOWLIST,
    NUMERIC_ALLOWLIST,
    TARGET_COLUMN,
    build_feature_frame,
    select_model_columns,
)


def test_select_model_columns_returns_existing_allowlisted_columns_only():
    df = pd.DataFrame(
        {
            TARGET_COLUMN: [100000, 120000],
            "Mileage_km": [10000, 20000],
            "Model": ["911", "Cayenne"],
            "unused": [1, 2],
        }
    )

    selected = select_model_columns(df)

    assert selected.target == TARGET_COLUMN
    assert selected.numeric == ("Mileage_km",)
    assert selected.categorical == ("Model",)
    assert selected.features == ["Mileage_km", "Model"]


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
        }
    )

    selected = select_model_columns(df)

    assert selected.numeric == ("Mileage_km", "log_mileage", "Mileage_sq")
    assert selected.categorical == (
        "Transmission",
        "Drive",
        "Ready to drive",
        "Car location",
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
    with pytest.raises(AttributeError):
        selected.numeric.append("extra")


def test_allowlists_are_immutable_tuples():
    assert isinstance(NUMERIC_ALLOWLIST, tuple)
    assert isinstance(CATEGORICAL_ALLOWLIST, tuple)


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
    assert selected.features == ["Mileage_km", "Model"]


def test_build_feature_frame_raises_for_missing_target():
    df = pd.DataFrame({"Mileage_km": [10000]})

    with pytest.raises(ValueError, match="target"):
        build_feature_frame(df)


def test_build_feature_frame_raises_for_no_supported_features():
    df = pd.DataFrame({TARGET_COLUMN: [100000]})

    with pytest.raises(ValueError, match="supported features"):
        build_feature_frame(df)
