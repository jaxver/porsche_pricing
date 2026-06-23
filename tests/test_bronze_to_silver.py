import pandas as pd
import pytest

from elferspot_listings.data_processing.bronze_to_silver import (
    clean_mileage,
    convert_prices_to_eur,
    create_derived_features,
)


def test_clean_mileage_handles_km_miles_malformed_and_restored_rows():
    df = pd.DataFrame(
        {
            "Mileage": ["10,000 mi", "20,000 km", "about 30k", None],
            "Condition": ["Used", "Used", "Used", "Fully restored"],
        }
    )

    result = clean_mileage(df)

    assert len(result) == 3
    assert result["Mileage_km"].tolist() == pytest.approx([16093.4, 20000.0, 1.0])


def test_convert_prices_to_eur_uses_supplied_rates():
    df = pd.DataFrame(
        {
            "price": [100000, 100000, 100000],
            "currency": ["EUR", "USD", "GBP"],
        }
    )

    result = convert_prices_to_eur(df, rates={"EUR": 1.0, "USD": 0.9, "GBP": 1.2})

    assert result["price_in_eur"].tolist() == [100000.0, 90000.0, 120000.0]


def test_create_derived_features_flags_known_owners_and_restoration():
    df = pd.DataFrame(
        {
            "Number of vehicle owners": ["Unknown", "2"],
            "Condition": ["Used", "Fully restored"],
        }
    )

    result = create_derived_features(df)

    assert result["owners_known"].tolist() == [0, 1]
    assert result["is_fully_restored"].tolist() == [0, 1]
