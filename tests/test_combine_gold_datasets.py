from __future__ import annotations

from pathlib import Path

import pandas as pd

from elferspot_listings.data_processing.combine_gold_datasets import combine_gold_datasets, score_row_completeness


def test_score_row_completeness_weights_key_columns():
    df = pd.DataFrame(
        {
            "URL": ["a", "b"],
            "price_in_eur": [100000, None],
            "Mileage_km": [10000, None],
            "Description": ["full", None],
            "Scraped_At": ["2026-01-01", None],
        }
    )

    scores = score_row_completeness(df)

    assert scores.iloc[0] > scores.iloc[1]


def test_combine_gold_datasets_keeps_more_complete_row_per_url(tmp_path):
    old_path = tmp_path / "old.xlsx"
    new_path = tmp_path / "new.xlsx"
    output_path = tmp_path / "combined.xlsx"

    old_df = pd.DataFrame(
        {
            "URL": ["https://example.com/1", "https://example.com/2"],
            "Title": ["old-a", "old-b"],
            "Description": ["short", "old desc"],
            "Scraped_At": ["2026-01-01", "2026-01-01"],
            "price_in_eur": [100000.0, 200000.0],
            "Mileage_km": [10000.0, 20000.0],
        }
    )
    new_df = pd.DataFrame(
        {
            "URL": ["https://example.com/1", "https://example.com/3"],
            "Title": ["new-a", "new-c"],
            "Description": ["much more complete description", "new desc"],
            "Scraped_At": ["2026-02-01", "2026-02-01"],
            "price_in_eur": [100500.0, 300000.0],
            "Mileage_km": [9999.0, 30000.0],
            "Extra": ["x", "y"],
        }
    )

    old_df.to_excel(old_path, index=False)
    new_df.to_excel(new_path, index=False)

    combined = combine_gold_datasets(old_path, new_path, output_path)

    assert output_path.exists()
    assert len(combined) == 3
    chosen = combined.loc[combined["URL"] == "https://example.com/1"].iloc[0]
    assert chosen["Title"] == "new-a"
    assert chosen["Description"] == "much more complete description"
    pd.testing.assert_frame_equal(pd.read_excel(old_path), old_df, check_dtype=False)
    pd.testing.assert_frame_equal(pd.read_excel(new_path), new_df, check_dtype=False)
