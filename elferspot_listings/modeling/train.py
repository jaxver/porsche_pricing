from __future__ import annotations

from dataclasses import dataclass
import logging
import json
import re
import shutil
import time
from pathlib import Path
from typing import Any

import config
import pandas as pd
from sklearn.model_selection import train_test_split

from elferspot_listings.evaluation.metrics import regression_metrics
from elferspot_listings.evaluation.reports import write_benchmark_report

from . import benchmark_db
from .baselines import (
    MedianRegressor,
    build_elasticnet_pipeline,
    build_perpetual_pipeline,
    build_ridge_pipeline,
    build_skrub_ridge_pipeline,
    build_xgboost_pipeline,
)
from .catboost_model import fit_catboost_regressor, predict_catboost_eur, save_catboost_model
from .challengers import (
    OptionalDependencyNotInstalledError,
    run_autogluon_regression,
    run_tabpfn_client_regression,
    run_tabpfn_regression,
)
from .features import build_feature_frame
from .persistence import SkopsNotInstalledError, save_sklearn_model, write_model_card


logger = logging.getLogger(__name__)

SUPPORTED_MODEL_NAMES = {
    "median",
    "ridge",
    "elasticnet",
    "skrub_ridge",
    "xgboost",
    "perpetual",
    "catboost",
    "tabpfn",
    "autogluon",
    "all",
}
DEFAULT_MODEL_NAMES = ("median", "ridge", "elasticnet", "skrub_ridge")


def tune_elasticnet_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected: Any,
    random_state: int = 42,
    n_trials: int = 25,
) -> dict[str, float | int]:
    import optuna

    X_tune, X_valid, y_tune, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    def objective(trial: Any) -> float:
        params = {
            "alpha": trial.suggest_float("alpha", 1e-5, 10.0, log=True),
            "l1_ratio": trial.suggest_float("l1_ratio", 0.0, 1.0),
            "max_iter": 20000,
        }
        model = build_elasticnet_pipeline(selected, **params)
        _, metrics = _score_model(model, X_tune, y_tune, X_valid, y_valid)
        return metrics["mae_eur"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return {**study.best_params, "max_iter": 20000}


def tune_catboost_params(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    selected: Any,
    random_state: int = 42,
    n_trials: int = 25,
    device: str = "cpu",
    gpu_devices: str | None = None,
) -> dict[str, float | int]:
    import optuna

    X_tune, X_valid, y_tune, y_valid = train_test_split(X_train, y_train, test_size=0.25, random_state=random_state)

    def objective(trial: Any) -> float:
        params = {
            "iterations": trial.suggest_int("iterations", 250, 1200),
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
            "depth": trial.suggest_int("depth", 3, 8),
            "l2_leaf_reg": trial.suggest_float("l2_leaf_reg", 1.0, 20.0, log=True),
        }
        model = fit_catboost_regressor(
            X_tune,
            y_tune,
            selected,
            random_state=random_state,
            params=params,
            device=device,
            gpu_devices=gpu_devices,
        )
        _, metrics = _score_catboost_model(model, X_valid, y_valid)
        return metrics["mae_eur"]

    study = optuna.create_study(direction="minimize")
    study.optimize(objective, n_trials=n_trials)
    return study.best_params


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


def _score_catboost_model(model: Any, X_test: pd.DataFrame, y_test: pd.Series) -> tuple[pd.DataFrame, dict[str, float]]:
    predicted = predict_catboost_eur(model, X_test)

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


def _score_predictions(model_name: str, y_test: pd.Series, predicted: Any) -> tuple[pd.DataFrame, dict[str, float]]:
    predicted_values = pd.Series(predicted, index=y_test.index, dtype=float)
    predictions = pd.DataFrame(
        {
            "row_index": y_test.index,
            "actual_price_eur": y_test.to_numpy(dtype=float),
            "predicted_price_eur": predicted_values.to_numpy(dtype=float),
        }
    )
    predictions["model_name"] = model_name
    predictions["residual_eur"] = predictions["actual_price_eur"] - predictions["predicted_price_eur"]
    metrics = regression_metrics(y_test, predicted_values)
    return predictions, metrics


def _prepare_tabpfn_features(X_train: pd.DataFrame, X_test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train_encoded = pd.get_dummies(X_train, dummy_na=True)
    test_encoded = pd.get_dummies(X_test, dummy_na=True)

    train_fill_values = train_encoded.median(numeric_only=True).fillna(0)
    train_encoded = train_encoded.fillna(train_fill_values)

    test_encoded = test_encoded.reindex(columns=train_encoded.columns, fill_value=0)
    test_encoded = test_encoded.fillna(train_fill_values)

    return train_encoded, test_encoded


def _normalize_tabpfn_checkpoint_alias(model_path: str | None) -> tuple[str, str | None]:
    if model_path in (None, "default"):
        return "tabpfn_default", None
    if model_path == "mediumdata":
        return "tabpfn_mediumdata", "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"
    if model_path == "ood":
        return "tabpfn_ood", "tabpfn-v3-regressor-v3_20260506_ood.ckpt"
    if not model_path.endswith(".ckpt"):
        raise ValueError(
            f"Unsupported TabPFN checkpoint alias '{model_path}'. Supported aliases: None/default, mediumdata, ood, or a string ending in .ckpt."
        )
    safe_stem = re.sub(r"[^0-9A-Za-z]+", "_", Path(model_path).stem).strip("_") or "custom"
    return f"tabpfn_{safe_stem.lower()}", model_path


def _cleanup_autogluon_output(output_path: Path, autogluon_trained: bool) -> None:
    if autogluon_trained:
        return
    if output_path.is_dir():
        shutil.rmtree(output_path)
    elif output_path.exists():
        output_path.unlink()


def _save_sklearn_artifact(model_name: str, model: Any, artifacts_dir: Path, skipped_models: dict[str, str]) -> bool:
    artifact_path = artifacts_dir / f"{model_name}.skops"
    try:
        save_sklearn_model(model, artifact_path)
    except SkopsNotInstalledError:
        skipped_models[f"{model_name}_artifact"] = "skops is not installed"
        if artifact_path.exists():
            artifact_path.unlink()
        return False
    except Exception:
        if artifact_path.exists():
            artifact_path.unlink()
        raise
    return True


def _cleanup_stale_sklearn_artifacts(artifacts_dir: Path, saved_models: set[str]) -> None:
    for model_name in ("ridge", "elasticnet", "xgboost", "skrub_ridge", "perpetual"):
        if model_name in saved_models:
            continue
        artifact_path = artifacts_dir / f"{model_name}.skops"
        if artifact_path.exists():
            artifact_path.unlink()


def _cleanup_written_sklearn_artifacts(artifact_paths: list[Path]) -> None:
    for artifact_path in artifact_paths:
        if artifact_path.exists():
            artifact_path.unlink()


def _normalize_requested_models(models: list[str] | None) -> set[str] | None:
    if models is None:
        return None
    invalid_models = sorted({model_name for model_name in models if model_name not in SUPPORTED_MODEL_NAMES})
    if invalid_models:
        raise ValueError(
            f"Unsupported model names: {', '.join(invalid_models)}. Supported models: {', '.join(sorted(SUPPORTED_MODEL_NAMES))}"
        )
    return set(models)


def _should_run_model(requested_models: set[str] | None, model_name: str, legacy_enabled: bool) -> bool:
    if requested_models is None:
        return legacy_enabled
    if "all" in requested_models:
        return model_name in DEFAULT_MODEL_NAMES or model_name in requested_models or legacy_enabled
    return model_name in requested_models


def _benchmark_metrics_for_db(metrics: dict[str, dict[str, float]]) -> dict[str, dict[str, float]]:
    return {
        model_name: {
            "mae_eur": values["mae_eur"],
            "median_ae": values["median_ae_eur"],
            "mape": values["mape"],
            "within_10": values["within_10pct"],
            "within_15": values["within_15pct"],
        }
        for model_name, values in metrics.items()
    }


def _log_benchmark_run(
    *,
    output_path: Path,
    random_state: int,
    train_catboost: bool,
    run_tabpfn: bool,
    run_autogluon: bool,
    autogluon_time_limit: int,
    metrics: dict[str, dict[str, float]],
    skipped_models: dict[str, str],
    start_time: float,
) -> None:
    try:
        run_id = benchmark_db.insert_run(
            config.BENCHMARK_DB,
            random_state=random_state,
            train_catboost=train_catboost,
            run_tabpfn=run_tabpfn,
            run_autogluon=run_autogluon,
            autogluon_tl=autogluon_time_limit,
            output_dir=output_path,
            duration_sec=time.perf_counter() - start_time,
        )
        benchmark_db.insert_metrics(config.BENCHMARK_DB, run_id, _benchmark_metrics_for_db(metrics))
        benchmark_db.insert_skipped(config.BENCHMARK_DB, run_id, skipped_models)
    except Exception:
        logger.exception("benchmark DB logging failed")


def train_baseline_models(
    gold_df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    train_catboost: bool = False,
    tune_elasticnet: bool = False,
    tune_catboost: bool = False,
    tuning_trials: int = 25,
    run_xgboost: bool = False,
    run_perpetual: bool = False,
    run_tabpfn: bool = False,
    tabpfn_model_paths: list[str | None] | None = None,
    tabpfn_backend: str = "local",
    tabpfn_thinking: bool = False,
    tabpfn_thinking_effort: str = "medium",
    tabpfn_thinking_timeout: float | int | None = None,
    tabpfn_thinking_metric: str = "rmse",
    run_autogluon: bool = False,
    autogluon_time_limit: int = 600,
    autogluon_presets: str = "best_quality",
    autogluon_dynamic_stacking: str = "auto",
    autogluon_clean_output: bool = False,
    models: list[str] | None = None,
    device: str = "cpu",
    gpu_devices: str | None = None,
) -> BenchmarkResult:
    start_time = time.perf_counter()
    requested_models = _normalize_requested_models(models)
    X, y, selected = build_feature_frame(gold_df)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=random_state)

    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    metrics: dict[str, dict[str, float]] = {}
    skipped_models: dict[str, str] = {}
    prediction_frames: list[pd.DataFrame] = []
    artifacts_dir = output_path / "artifacts"
    catboost_artifact_path = output_path / "artifacts" / "catboost.cbm"
    catboost_trained = False
    baseline_artifact_models: dict[str, Any] = {}
    saved_artifact_models: set[str] = set()
    saved_artifact_paths: list[Path] = []

    elasticnet_params = dict(config.MODEL_CONFIG["elasticnet"])
    if _should_run_model(requested_models, "elasticnet", legacy_enabled=True) and tune_elasticnet:
        elasticnet_params = tune_elasticnet_params(X_train, y_train, selected, random_state=random_state, n_trials=tuning_trials)

    should_run_tabpfn = _should_run_model(
        requested_models,
        "tabpfn",
        legacy_enabled=run_tabpfn or tabpfn_model_paths is not None,
    )

    if should_run_tabpfn and tabpfn_backend == "client" and tabpfn_model_paths is not None:
        raise ValueError("TabPFN checkpoints are local-backend only. Remove --tabpfn-checkpoint when using --tabpfn-backend client.")
    if should_run_tabpfn and tabpfn_backend == "local" and tabpfn_thinking:
        raise ValueError("TabPFN thinking mode requires the client backend.")

    if _should_run_model(requested_models, "median", legacy_enabled=True):
        model = MedianRegressor()
        model_predictions, model_metrics = _score_model(model, X_train, y_train, X_test, y_test)
        model_predictions = model_predictions.assign(model_name="median")
        metrics["median"] = model_metrics
        prediction_frames.append(model_predictions)

    if _should_run_model(requested_models, "ridge", legacy_enabled=True):
        ridge_model = build_ridge_pipeline(selected)
        model_predictions, model_metrics = _score_model(ridge_model, X_train, y_train, X_test, y_test)
        model_predictions = model_predictions.assign(model_name="ridge")
        metrics["ridge"] = model_metrics
        prediction_frames.append(model_predictions)
        baseline_artifact_models["ridge"] = ridge_model

    if _should_run_model(requested_models, "elasticnet", legacy_enabled=True):
        elasticnet_model = build_elasticnet_pipeline(
            selected,
            alpha=float(elasticnet_params["alpha"]),
            l1_ratio=float(elasticnet_params["l1_ratio"]),
            max_iter=int(elasticnet_params["max_iter"]),
        )
        model_predictions, model_metrics = _score_model(elasticnet_model, X_train, y_train, X_test, y_test)
        model_predictions = model_predictions.assign(model_name="elasticnet")
        metrics["elasticnet"] = model_metrics
        prediction_frames.append(model_predictions)
        baseline_artifact_models["elasticnet"] = elasticnet_model

    if _should_run_model(requested_models, "skrub_ridge", legacy_enabled=True):
        try:
            skrub_model = build_skrub_ridge_pipeline(selected)
        except ImportError:
            skipped_models["skrub_ridge"] = "skrub is not installed"
        else:
            model_predictions, model_metrics = _score_model(skrub_model, X_train, y_train, X_test, y_test)
            model_predictions = model_predictions.assign(model_name="skrub_ridge")
            metrics["skrub_ridge"] = model_metrics
            prediction_frames.append(model_predictions)
            baseline_artifact_models["skrub_ridge"] = skrub_model

    if _should_run_model(requested_models, "xgboost", legacy_enabled=run_xgboost):
        try:
            xgboost_model = build_xgboost_pipeline(
                selected,
                random_state=random_state,
                **({"device": device} if device == "gpu" else {}),
            )
        except ImportError:
            skipped_models["xgboost"] = "xgboost is not installed"
        else:
            model_predictions, model_metrics = _score_model(xgboost_model, X_train, y_train, X_test, y_test)
            model_predictions = model_predictions.assign(model_name="xgboost")
            metrics["xgboost"] = model_metrics
            prediction_frames.append(model_predictions)
            baseline_artifact_models["xgboost"] = xgboost_model

    if _should_run_model(requested_models, "perpetual", legacy_enabled=run_perpetual):
        try:
            perpetual_model = build_perpetual_pipeline(selected, random_state=random_state)
        except ImportError:
            skipped_models["perpetual"] = "perpetual is not installed"
        else:
            model_predictions, model_metrics = _score_model(perpetual_model, X_train, y_train, X_test, y_test)
            model_predictions = model_predictions.assign(model_name="perpetual")
            metrics["perpetual"] = model_metrics
            prediction_frames.append(model_predictions)
            baseline_artifact_models["perpetual"] = perpetual_model

    tabpfn_ran = False
    if should_run_tabpfn and tabpfn_backend == "client":
        model_name = "tabpfn_client_thinking" if tabpfn_thinking else "tabpfn_client"
        try:
            _, tabpfn_predictions, metadata = run_tabpfn_client_regression(
                X_train,
                y_train,
                X_test,
                random_state=random_state,
                thinking_mode=tabpfn_thinking,
                thinking_effort=tabpfn_thinking_effort,
                thinking_metric=tabpfn_thinking_metric,
                thinking_timeout_s=tabpfn_thinking_timeout,
            )
        except OptionalDependencyNotInstalledError as exc:
            skipped_models[model_name] = str(exc)
        else:
            tabpfn_ran = True
            model_name = str(metadata.get("model_name", model_name))
            model_predictions, model_metrics = _score_predictions(model_name, y_test, tabpfn_predictions)
            metrics[model_name] = model_metrics
            prediction_frames.append(model_predictions)
    else:
        tabpfn_requests = tabpfn_model_paths if should_run_tabpfn and tabpfn_model_paths is not None else ([None] if should_run_tabpfn else [])
        if tabpfn_requests:
            tabpfn_X_train, tabpfn_X_test = _prepare_tabpfn_features(X_train, X_test)
            missing_tabpfn_message: str | None = None
            for requested_model_path in tabpfn_requests:
                model_name, normalized_model_path = _normalize_tabpfn_checkpoint_alias(requested_model_path)
                if missing_tabpfn_message is not None:
                    skipped_models[model_name] = missing_tabpfn_message
                    continue
                try:
                    _, tabpfn_predictions, _ = run_tabpfn_regression(
                        tabpfn_X_train,
                        y_train,
                        tabpfn_X_test,
                        random_state=random_state,
                        model_path=normalized_model_path,
                        model_name=model_name,
                        **({"device": device, "gpu_devices": gpu_devices} if device == "gpu" else {}),
                    )
                except OptionalDependencyNotInstalledError as exc:
                    missing_tabpfn_message = str(exc)
                    skipped_models[model_name] = missing_tabpfn_message
                else:
                    tabpfn_ran = True
                    model_predictions, model_metrics = _score_predictions(model_name, y_test, tabpfn_predictions)
                    metrics[model_name] = model_metrics
                    prediction_frames.append(model_predictions)

    if _should_run_model(requested_models, "autogluon", legacy_enabled=run_autogluon):
        autogluon_train_df = X_train.copy()
        autogluon_train_df["price_in_eur"] = y_train.to_numpy(dtype=float)
        autogluon_test_df = X_test.copy()
        autogluon_test_df["price_in_eur"] = y_test.to_numpy(dtype=float)
        try:
            _, autogluon_predictions, _, _ = run_autogluon_regression(
                autogluon_train_df,
                autogluon_test_df,
                "price_in_eur",
                output_path,
                time_limit=autogluon_time_limit,
                presets=autogluon_presets,
                dynamic_stacking=autogluon_dynamic_stacking,
                clean_output=autogluon_clean_output,
            )
        except OptionalDependencyNotInstalledError as exc:
            skipped_models["autogluon"] = str(exc)
        else:
            model_predictions, model_metrics = _score_predictions("autogluon", y_test, autogluon_predictions)
            metrics["autogluon"] = model_metrics
            prediction_frames.append(model_predictions)

    if _should_run_model(requested_models, "catboost", legacy_enabled=train_catboost):
        try:
            catboost_params = None
            if _should_run_model(requested_models, "catboost", legacy_enabled=train_catboost) and tune_catboost:
                catboost_params = tune_catboost_params(
                    X_train,
                    y_train,
                    selected,
                    random_state=random_state,
                    n_trials=tuning_trials,
                    **({"device": device, "gpu_devices": gpu_devices} if device == "gpu" else {}),
                )
            catboost_model = fit_catboost_regressor(
                X_train,
                y_train,
                selected,
                random_state=random_state,
                params=catboost_params,
                **({"device": device, "gpu_devices": gpu_devices} if device == "gpu" else {}),
            )
        except ImportError:
            skipped_models["catboost"] = "catboost is not installed"
        else:
            model_predictions, model_metrics = _score_catboost_model(catboost_model, X_test, y_test)
            model_predictions = model_predictions.assign(model_name="catboost")
            metrics["catboost"] = model_metrics
            prediction_frames.append(model_predictions)
            save_catboost_model(catboost_model, catboost_artifact_path)
            catboost_trained = True

    if not catboost_trained and catboost_artifact_path.exists():
        catboost_artifact_path.unlink()

    _cleanup_autogluon_output(output_path / "autogluon", "autogluon" in metrics)

    try:
        for model_name, model in baseline_artifact_models.items():
            artifact_path = artifacts_dir / f"{model_name}.skops"
            if _save_sklearn_artifact(model_name, model, artifacts_dir, skipped_models):
                saved_artifact_models.add(model_name)
                saved_artifact_paths.append(artifact_path)
    except Exception:
        _cleanup_written_sklearn_artifacts(saved_artifact_paths)
        raise

    _cleanup_stale_sklearn_artifacts(artifacts_dir, saved_artifact_models)

    card_candidates = [name for name in ("ridge", "skrub_ridge", "median") if name in metrics]
    if card_candidates:
        card_model_name = min(card_candidates, key=lambda name: metrics[name]["mae_eur"])
        card_metrics: dict[str, float] = metrics[card_model_name]
    else:
        card_model_name = next(iter(metrics), "unknown_model")
        card_metrics = metrics.get(card_model_name, {})

    write_model_card(
        output_path / "MODEL_CARD.md",
        {
            "model_name": card_model_name,
            "purpose": "Predict Porsche listing prices from cleaned Gold-layer listing data.",
            "target": "price_in_eur",
            "metrics": card_metrics,
            "limitations": [
                "Synthetic benchmark data does not represent production market breadth.",
                "Artifacts are only written when skops is available.",
            ],
            "usage_notes": [
                "Use the benchmark report and holdout predictions alongside the artifact.",
                "Load `.skops` artifacts with `skops.io.load` after inspecting untrusted types.",
            ],
        },
    )

    if prediction_frames:
        predictions = pd.concat(prediction_frames, ignore_index=True)
    else:
        predictions = pd.DataFrame(
            columns=[
                "row_index",
                "model_name",
                "actual_price_eur",
                "predicted_price_eur",
                "residual_eur",
            ]
        )
    predictions = predictions[["row_index", "model_name", "actual_price_eur", "predicted_price_eur", "residual_eur"]]
    predictions.to_csv(output_path / "predictions.csv", index=False)

    write_benchmark_report(metrics, output_path)

    _log_benchmark_run(
        output_path=output_path,
        random_state=random_state,
        train_catboost=train_catboost,
        run_tabpfn=tabpfn_ran,
        run_autogluon=run_autogluon,
        autogluon_time_limit=autogluon_time_limit,
        metrics=metrics,
        skipped_models=skipped_models,
        start_time=start_time,
    )

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
