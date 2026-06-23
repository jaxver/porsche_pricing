"""Regression evaluation metrics for Porsche price benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Compute a compact regression summary in EUR-space."""
    actual = pd.Series(y_true, dtype="float64")
    predicted = pd.Series(y_pred, dtype="float64")
    errors = actual - predicted
    abs_errors = errors.abs()

    nonzero_actual = actual != 0
    pct_errors = pd.Series(np.zeros(len(actual), dtype="float64"), index=actual.index)
    pct_errors.loc[nonzero_actual] = abs_errors.loc[nonzero_actual] / actual.loc[nonzero_actual].abs()

    return {
        "mae_eur": float(abs_errors.mean()),
        "median_ae_eur": float(abs_errors.median()),
        "mape": float(pct_errors.loc[nonzero_actual].mean() if nonzero_actual.any() else 0.0),
        "within_10pct": float((pct_errors <= 0.10).mean()),
        "within_15pct": float((pct_errors <= 0.15).mean()),
    }


def segment_metrics(
    df: pd.DataFrame,
    actual_col: str,
    predicted_col: str,
    segment_cols: Iterable[str],
) -> pd.DataFrame:
    """Compute metrics per value of each available segment column."""
    rows: list[dict[str, float | int | str]] = []

    for segment_col in segment_cols:
        if segment_col not in df.columns:
            continue

        for segment_value in sorted(df[segment_col].dropna().unique(), key=str):
            subset = df[df[segment_col] == segment_value]
            metrics = regression_metrics(subset[actual_col], subset[predicted_col])
            rows.append(
                {
                    "segment_column": segment_col,
                    "segment_value": segment_value,
                    "n_rows": int(len(subset)),
                    **metrics,
                }
            )

    return pd.DataFrame(rows)
