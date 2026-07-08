from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "price_in_eur"

NUMERIC_ALLOWLIST = (
    "Mileage_km",
    "Year of construction",
    "listing_score",
    "owners_known",
    "is_fully_restored",
    "Paint-to-Sample (PTS)",
    "log_mileage",
    "Mileage_sq",
    "model_cat_ordered",
    "inv_mileage",
    "Mileage_model_cat",
    "inv_Mileage_model_cat",
    "Mileage_sq_model_cat",
    "is_rare",
    "is_restomod",
    "is_race_ready",
    "restoration_full",
    "restoration_partial",
    "has_docs",
    "is_matching_numbers",
    "is_mint",
    "is_accident_free",
    "has_upgrades",
    "first_owner",
    "state_yes",
    "state_Rear drive",
    "matching_yes",
    "price_inflation_factor",
)

CATEGORICAL_ALLOWLIST = (
    "Model",
    "Series",
    "model_category",
    "Condition",
    "Matching numbers",
    "Interior color",
    "Exterior color",
    "Country",
    "Transmission",
    "Drive",
    "Ready to drive",
    "Car location",
)


@dataclass(frozen=True)
class SelectedColumns:
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def select_model_columns(df: pd.DataFrame) -> SelectedColumns:
    numeric = tuple(column for column in NUMERIC_ALLOWLIST if column in df.columns)
    categorical = tuple(column for column in CATEGORICAL_ALLOWLIST if column in df.columns)
    return SelectedColumns(target=TARGET_COLUMN, numeric=numeric, categorical=categorical)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, SelectedColumns]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    selected = select_model_columns(df)
    if not selected.features:
        raise ValueError("No supported features found in DataFrame")

    modeled = df.dropna(subset=[TARGET_COLUMN]).copy()
    X = modeled.loc[:, list(selected.features)].copy()
    for col in selected.categorical:
        if col in X.columns:
            X.loc[:, col] = X[col].fillna("Unknown")
    y = modeled[TARGET_COLUMN].astype(float)
    return X, y, selected  # type: ignore[return-value]
