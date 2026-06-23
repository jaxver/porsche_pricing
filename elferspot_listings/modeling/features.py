from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "price_in_eur"

NUMERIC_ALLOWLIST = [
    "Mileage_km",
    "Year of construction",
    "listing_score",
    "owners_known",
    "is_fully_restored",
    "Paint-to-Sample (PTS)",
]

CATEGORICAL_ALLOWLIST = [
    "Model",
    "Series",
    "model_category",
    "Condition",
    "Matching numbers",
    "Interior color",
    "Exterior color",
    "Country",
]


@dataclass(frozen=True)
class SelectedColumns:
    target: str
    numeric: list[str]
    categorical: list[str]

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def select_model_columns(df: pd.DataFrame) -> SelectedColumns:
    numeric = [column for column in NUMERIC_ALLOWLIST if column in df.columns]
    categorical = [column for column in CATEGORICAL_ALLOWLIST if column in df.columns]
    return SelectedColumns(target=TARGET_COLUMN, numeric=numeric, categorical=categorical)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, SelectedColumns]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    selected = select_model_columns(df)
    if not selected.features:
        raise ValueError("No supported features found in DataFrame")

    modeled = df.dropna(subset=[TARGET_COLUMN]).copy()
    X = modeled[selected.features].copy()
    y = modeled[TARGET_COLUMN].astype(float)
    return X, y, selected
