from __future__ import annotations

from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from .features import SelectedColumns


def default_catboost_params(random_state: int = 42) -> dict[str, Any]:
    return {
        "loss_function": "RMSE",
        "iterations": 1000,
        "depth": 6,
        "learning_rate": 0.05,
        "random_seed": random_state,
        "verbose": False,
        "allow_writing_files": False,
    }


def _gpu_catboost_params(device: str = "cpu", gpu_devices: str | None = None) -> dict[str, Any]:
    if device != "gpu":
        return {}

    params: dict[str, Any] = {"task_type": "GPU"}
    if gpu_devices is not None:
        params["devices"] = gpu_devices
    return params


def _prepare_catboost_frame(X, selected: SelectedColumns) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.DataFrame(X).copy()
    categorical_columns = [column for column in selected.categorical if column in frame.columns]
    for col in categorical_columns:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    cat_features = [cast(int, frame.columns.get_loc(column)) for column in categorical_columns]
    return frame, cat_features


def fit_catboost_regressor(
    X_train,
    y_train,
    selected: SelectedColumns,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    gpu_devices: str | None = None,
):
    from catboost import CatBoostRegressor, Pool

    frame, cat_features = _prepare_catboost_frame(X_train, selected)
    target = np.asarray(y_train, dtype=float)
    if np.any(target <= 0):
        raise ValueError("Target values must be positive before applying the log transform")

    train_pool = Pool(frame, label=np.log(target), cat_features=cat_features)
    model_params = default_catboost_params(random_state=random_state)
    if params:
        model_params.update(params)
    model_params.update(_gpu_catboost_params(device=device, gpu_devices=gpu_devices))
    model = CatBoostRegressor(**model_params)
    model.fit(train_pool)
    return model


def predict_catboost_eur(model, X):
    return np.exp(np.asarray(model.predict(X), dtype=float))


def fit_catboost_quantile_interval(
    X_train,
    y_train,
    selected: SelectedColumns,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    gpu_devices: str | None = None,
) -> dict[str, Any]:
    from catboost import CatBoostRegressor, Pool

    frame, cat_features = _prepare_catboost_frame(X_train, selected)
    target = np.asarray(y_train, dtype=float)
    if np.any(target <= 0):
        raise ValueError("Target values must be positive before applying the log transform")

    train_pool = Pool(frame, label=np.log(target), cat_features=cat_features)
    base_params = default_catboost_params(random_state=random_state)
    if params:
        base_params.update(params)
    base_params.update(_gpu_catboost_params(device=device, gpu_devices=gpu_devices))

    fitted: dict[str, Any] = {}
    for name, alpha in (("lower", 0.05), ("median", 0.5), ("upper", 0.95)):
        model_params = {**base_params, "loss_function": f"Quantile:alpha={alpha}"}
        model = CatBoostRegressor(**model_params)
        model.fit(train_pool)
        fitted[name] = model
    fitted["_selected"] = selected
    return fitted


def predict_catboost_interval_eur(interval_models: dict[str, Any], X) -> pd.DataFrame:
    selected = interval_models.get("_selected")
    frame = pd.DataFrame(X).copy() if selected is None else _prepare_catboost_frame(X, selected)[0]
    predictions = pd.DataFrame(index=frame.index)
    predictions["pred_lower"] = np.exp(np.asarray(interval_models["lower"].predict(frame), dtype=float))
    predictions["pred_price"] = np.exp(np.asarray(interval_models["median"].predict(frame), dtype=float))
    predictions["pred_upper"] = np.exp(np.asarray(interval_models["upper"].predict(frame), dtype=float))
    ordered = np.sort(predictions[["pred_lower", "pred_price", "pred_upper"]].to_numpy(dtype=float), axis=1)
    predictions.loc[:, ["pred_lower", "pred_price", "pred_upper"]] = ordered
    return predictions


def save_catboost_model(model, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
