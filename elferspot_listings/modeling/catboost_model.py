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


def fit_catboost_regressor(X_train, y_train, selected: SelectedColumns, random_state: int = 42):
    from catboost import CatBoostRegressor, Pool

    frame = pd.DataFrame(X_train).copy()
    target = np.asarray(y_train, dtype=float)
    if np.any(target <= 0):
        raise ValueError("Target values must be positive before applying the log transform")

    categorical_columns = [column for column in selected.categorical if column in frame.columns]
    cat_features = [frame.columns.get_loc(column) for column in categorical_columns]

    train_pool = Pool(frame, label=np.log(target), cat_features=cat_features)
    model = CatBoostRegressor(**default_catboost_params(random_state=random_state))
    model.fit(train_pool)
    return model


def predict_catboost_eur(model, X):
    return np.exp(np.asarray(model.predict(X), dtype=float))


def save_catboost_model(model, path):
    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    model.save_model(output_path)
