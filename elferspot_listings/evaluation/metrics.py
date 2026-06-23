"""Regression evaluation metrics for Porsche price benchmarks."""

from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd


def regression_metrics(y_true: Iterable[float], y_pred: Iterable[float]) -> dict[str, float]:
    """Compute a compact regression summary in EUR-space."""
    actual = np.asarray(y_true, dtype=float).reshape(-1)
    predicted = np.asarray(y_pred, dtype=float).reshape(-1)

    if actual.size == 0 or predicted.size == 0:
        raise ValueError("y_true and y_pred must not be empty")
    if actual.shape != predicted.shape:
        raise ValueError("y_true and y_pred must have equal length")

    abs_errors = np.abs(actual - predicted)
    pct_errors = np.empty_like(abs_errors, dtype=float)
    zero_actual = actual == 0
    nonzero_actual = ~zero_actual
    pct_errors[nonzero_actual] = abs_errors[nonzero_actual] / np.abs(actual[nonzero_actual])
    pct_errors[zero_actual] = np.where(abs_errors[zero_actual] == 0, 0.0, np.inf)

    return {
        "mae_eur": float(abs_errors.mean()),
        "median_ae_eur": float(np.median(abs_errors)),
        "mape": float(np.mean(pct_errors)),
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
