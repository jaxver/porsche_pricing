from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

TARGET_COLUMN = "price_in_eur"
TEXT_SOURCE_COLUMNS = ("Title", "Model", "Description", "Secondary_Description")

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

TEXT_ALLOWLIST = (
    "listing_text",
    "Description",
)


@dataclass(frozen=True)
class SelectedColumns:
    target: str
    numeric: tuple[str, ...]
    categorical: tuple[str, ...]
    text: tuple[str, ...] = ()

    @property
    def features(self) -> list[str]:
        return [*self.numeric, *self.categorical, *self.text]

    @property
    def non_text_features(self) -> list[str]:
        return [*self.numeric, *self.categorical]


def add_listing_text_feature(df: pd.DataFrame) -> pd.DataFrame:
    if "listing_text" in df.columns:
        return df
    source_columns = [column for column in TEXT_SOURCE_COLUMNS if column in df.columns]
    if not source_columns:
        return df
    result = df.copy()
    result["listing_text"] = result[source_columns].fillna("").astype(str).agg(" ".join, axis=1)
    return result


def select_model_columns(df: pd.DataFrame) -> SelectedColumns:
    numeric = tuple(column for column in NUMERIC_ALLOWLIST if column in df.columns)
    categorical = tuple(column for column in CATEGORICAL_ALLOWLIST if column in df.columns)
    text = tuple(column for column in TEXT_ALLOWLIST if column in df.columns)
    if "listing_text" in text:
        text = ("listing_text",)
    return SelectedColumns(target=TARGET_COLUMN, numeric=numeric, categorical=categorical, text=text)


def build_feature_frame(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series, SelectedColumns]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"Missing required target column: {TARGET_COLUMN}")

    modeled = add_listing_text_feature(df.dropna(subset=[TARGET_COLUMN]).copy())
    selected = select_model_columns(modeled)
    if not selected.features:
        raise ValueError("No supported features found in DataFrame")

    X = modeled.loc[:, list(selected.features)].copy()
    for col in selected.categorical:
        if col in X.columns:
            X.loc[:, col] = X[col].fillna("Unknown")
    y = modeled[TARGET_COLUMN].astype(float)
    return X, y, selected  # type: ignore[return-value]
