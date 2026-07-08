from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import pandas as pd

from .features import CATEGORICAL_ALLOWLIST, TARGET_COLUMN


def screen_features(
    df: pd.DataFrame,
    *,
    columns: Sequence[str] | None = None,
    target_col: str = TARGET_COLUMN,
    max_null_fraction: float = 0.4,
    max_cardinality: int = 50,
    min_frequency: int = 2,
    categorical_columns: Sequence[str] = CATEGORICAL_ALLOWLIST,
) -> dict[str, Any]:
    """Return a structural feature-quality report with explicit exclusion reasons."""
    if target_col not in df.columns:
        raise ValueError(f"Missing target column: {target_col}")

    candidate_columns = list(columns) if columns is not None else [column for column in df.columns if column != target_col]
    missing = [column for column in candidate_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing feature columns: {missing}")

    target = pd.to_numeric(df[target_col], errors="coerce")
    target_violations: list[str] = []
    if target.isna().any():
        target_violations.append("target_has_nulls")
    if (target.dropna() <= 0).any():
        target_violations.append("target_has_non_positive_values")

    selected_features: list[str] = []
    excluded_features: list[dict[str, Any]] = []
    categorical_set = set(categorical_columns)

    for column in candidate_columns:
        series = df[column]
        reasons: list[str] = []
        null_fraction = float(series.isna().mean())
        if null_fraction > max_null_fraction:
            reasons.append(f"null_fraction_gt_{max_null_fraction:g}")
        if series.nunique(dropna=True) <= 1:
            reasons.append("constant")
        if column in categorical_set or pd.api.types.is_object_dtype(series) or isinstance(series.dtype, pd.CategoricalDtype):
            value_counts = series.dropna().value_counts()
            cardinality = int(value_counts.size)
            if cardinality > max_cardinality:
                reasons.append(f"cardinality_gt_{max_cardinality:g}")
            if not value_counts.empty and int(value_counts.min()) < min_frequency:
                reasons.append(f"min_frequency_lt_{min_frequency:g}")

        if reasons:
            excluded_features.append({"feature": column, "reasons": reasons})
        else:
            selected_features.append(column)

    return {
        "target": target_col,
        "target_violations": target_violations,
        "selected_features": selected_features,
        "excluded_features": excluded_features,
        "settings": {
            "max_null_fraction": max_null_fraction,
            "max_cardinality": max_cardinality,
            "min_frequency": min_frequency,
        },
    }
