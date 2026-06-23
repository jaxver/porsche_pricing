from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any

import pandas as pd
from sklearn.model_selection import train_test_split

from elferspot_listings.evaluation.metrics import regression_metrics
from elferspot_listings.evaluation.reports import write_benchmark_report

from .baselines import MedianRegressor, build_ridge_pipeline, build_skrub_ridge_pipeline
from .features import build_feature_frame


@dataclass(frozen=True)
class BenchmarkResult:
    metrics: dict[str, dict[str, float]]
    predictions: pd.DataFrame
    output_dir: Path
    skipped_models: dict[str, str]


def _score_model(model: Any, X_train: pd.DataFrame, y_train: pd.Series, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    model.fit(X_train, y_train)
    predicted = model.predict(X_test)

    predictions = pd.DataFrame(
        {
            "row_index": X_test.index,
            "actual_price_eur": y_test.to_numpy(dtype=float),
            "predicted_price_eur": predicted,
        }
    )
    predictions["model_name"] = ""
    predictions["residual_eur"] = predictions["actual_price_eur"] - predictions["predicted_price_eur"]
    metrics = regression_metrics(y_test, predicted)
    return predictions, metrics


def train_baseline_models(
    gold_df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
) -> BenchmarkResult:
    X, y, selected = build_feature_frame(gold_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    skipped_models: dict[str, str] = {}
    prediction_frames: list[pd.DataFrame] = []

    for model_name, model in (
        ("median", MedianRegressor()),
        ("ridge", build_ridge_pipeline(selected)),
    ):
        model_predictions, model_metrics = _score_model(model, X_train, y_train, X_test, y_test)
        model_predictions = model_predictions.assign(model_name=model_name)
        metrics[model_name] = model_metrics
        prediction_frames.append(model_predictions)

    try:
        skrub_model = build_skrub_ridge_pipeline(selected)
    except ImportError:
        skipped_models["skrub_ridge"] = "skrub is not installed"
    else:
        model_predictions, model_metrics = _score_model(skrub_model, X_train, y_train, X_test, y_test)
        model_predictions = model_predictions.assign(model_name="skrub_ridge")
        metrics["skrub_ridge"] = model_metrics
        prediction_frames.append(model_predictions)

    predictions = pd.concat(prediction_frames, ignore_index=True)
    predictions = predictions[["row_index", "model_name", "actual_price_eur", "predicted_price_eur", "residual_eur"]]
    predictions.to_csv(output_path / "predictions.csv", index=False)

    write_benchmark_report(metrics, output_path)

    if skipped_models:
        skipped_path = output_path / "skipped_models.json"
        skipped_path.write_text(json.dumps(skipped_models, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    else:
        skipped_path = output_path / "skipped_models.json"
        if skipped_path.exists():
            skipped_path.unlink()

    return BenchmarkResult(
        metrics=metrics,
        predictions=predictions,
        output_dir=output_path,
        skipped_models=skipped_models,
    )
