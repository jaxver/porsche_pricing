from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import SelectedColumns


class MedianRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X: Any, y: Any):
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any):
        if not hasattr(self, "median_"):
            raise AttributeError("MedianRegressor is not fitted yet")
        return np.full(shape=len(X), fill_value=self.median_, dtype=float)


def _build_feature_transformer(selected: SelectedColumns) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
        ]
    )

    transformers = []
    if selected.numeric:
        transformers.append(("numeric", numeric_pipeline, list(selected.numeric)))
    if selected.categorical:
        transformers.append(("categorical", categorical_pipeline, list(selected.categorical)))

    if not transformers:
        raise ValueError("SelectedColumns must include at least one feature column")

    return ColumnTransformer(
        transformers=transformers,
        remainder="drop",
    )


def _select_columns(frame, columns):
    return frame.loc[:, columns]


def _positive_log_target(y):
    values = np.asarray(y, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Target values must be positive before applying the log transform")
    return np.log(values)


def _exp_target(y):
    return np.exp(np.asarray(y, dtype=float))


def build_ridge_pipeline(selected: SelectedColumns) -> TransformedTargetRegressor:
    model = Pipeline(
        steps=[
            ("features", _build_feature_transformer(selected)),
            ("ridge", Ridge()),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )


def build_elasticnet_pipeline(
    selected: SelectedColumns,
    alpha: float = 1.0,
    l1_ratio: float = 0.5,
    max_iter: int = 10000,
) -> TransformedTargetRegressor:
    model = Pipeline(
        steps=[
            ("features", _build_feature_transformer(selected)),
            (
                "elasticnet",
                ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=max_iter),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )


def build_xgboost_pipeline(selected: SelectedColumns, random_state: int = 42) -> TransformedTargetRegressor:
    try:
        from xgboost import XGBRegressor
    except ModuleNotFoundError as exc:
        raise ImportError("xgboost is not installed") from exc

    model = Pipeline(
        steps=[
            ("features", _build_feature_transformer(selected)),
            (
                "xgboost",
                XGBRegressor(
                    objective="reg:squarederror",
                    n_estimators=700,
                    max_depth=4,
                    learning_rate=0.05,
                    subsample=0.8,
                    colsample_bytree=0.8,
                    reg_alpha=0.0,
                    reg_lambda=1.0,
                    random_state=random_state,
                    tree_method="hist",
                    n_jobs=-1,
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )


def build_skrub_ridge_pipeline(selected: SelectedColumns) -> TransformedTargetRegressor:
    from skrub import TableVectorizer

    model = Pipeline(
        steps=[
            (
                "select",
                FunctionTransformer(_select_columns, kw_args={"columns": selected.features}),
            ),
            ("features", TableVectorizer()),
            ("imputer", SimpleImputer(strategy="median")),
            ("ridge", Ridge()),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )
