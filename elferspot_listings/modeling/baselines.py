from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
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

    return ColumnTransformer(
        transformers=[
            ("numeric", numeric_pipeline, list(selected.numeric)),
            ("categorical", categorical_pipeline, list(selected.categorical)),
        ],
        remainder="drop",
    )


def _select_columns(frame, columns):
    return frame.loc[:, columns]


def build_ridge_pipeline(selected: SelectedColumns) -> TransformedTargetRegressor:
    model = Pipeline(
        steps=[
            ("features", _build_feature_transformer(selected)),
            ("ridge", Ridge()),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
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
            ("ridge", Ridge()),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=np.log1p,
        inverse_func=np.expm1,
    )
