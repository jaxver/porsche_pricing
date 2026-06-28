from __future__ import annotations

from pathlib import Path
from typing import Any

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

    frame = pd.DataFrame(X_train).copy()
    target = np.asarray(y_train, dtype=float)
    if np.any(target <= 0):
        raise ValueError("Target values must be positive before applying the log transform")

    categorical_columns = [column for column in selected.categorical if column in frame.columns]
    for col in categorical_columns:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    cat_features = [frame.columns.get_loc(column) for column in categorical_columns]

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


def save_catboost_model(model, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
