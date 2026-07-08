from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler

from .features import SelectedColumns

LINEAR_NUMERIC_EXCLUDE = {
    "model_cat_ordered",
    "inv_mileage",
    "Mileage_model_cat",
    "inv_Mileage_model_cat",
    "Mileage_sq_model_cat",
    "restoration_full",
    "restoration_partial",
    "is_restomod",
    "has_docs",
    "is_matching_numbers",
    "is_mint",
    "is_race_ready",
    "is_rare",
    "is_accident_free",
    "has_upgrades",
    "first_owner",
    "state_yes",
    "state_Rear drive",
    "matching_yes",
    "limited_production",
    "racing_history",
    "specialist_build",
    "bespoke_exclusive",
    "zero_running_hours",
    "engine_transmission_rebuilt",
    "cup_clubsport",
    "heritage_special",
}


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


def _flatten_text(values):
    array = np.asarray(values, dtype=object)
    if array.ndim == 1:
        return np.asarray(["" if value is None else str(value) for value in array], dtype=object)
    return np.asarray([" ".join("" if item is None else str(item) for item in row) for row in array], dtype=object)


def _linear_selected_columns(selected: SelectedColumns) -> SelectedColumns:
    return SelectedColumns(
        target=selected.target,
        numeric=tuple(column for column in selected.numeric if column not in LINEAR_NUMERIC_EXCLUDE),
        categorical=selected.categorical,
        text=selected.text,
    )


def _build_text_feature_transformer(
    selected: SelectedColumns,
    *,
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 1,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
) -> ColumnTransformer:
    linear_selected = _linear_selected_columns(selected)
    transformers = []
    if linear_selected.non_text_features:
        tabular_selected = SelectedColumns(
            target=linear_selected.target,
            numeric=linear_selected.numeric,
            categorical=linear_selected.categorical,
        )
        transformers.extend(_build_feature_transformer(tabular_selected).transformers)
    if linear_selected.text:
        text_pipeline = Pipeline(
            steps=[
                ("imputer", SimpleImputer(strategy="constant", fill_value="")),
                ("flatten", FunctionTransformer(_flatten_text, validate=False)),
                (
                    "tfidf",
                    TfidfVectorizer(
                        max_features=tfidf_max_features,
                        min_df=tfidf_min_df,
                        ngram_range=tfidf_ngram_range,
                        sublinear_tf=True,
                    ),
                ),
            ]
        )
        transformers.append(("text", text_pipeline, list(linear_selected.text)))

    if not transformers:
        raise ValueError("SelectedColumns must include at least one feature column")

    return ColumnTransformer(transformers=transformers, remainder="drop", sparse_threshold=1.0)


def _select_columns(frame, columns):
    return frame.loc[:, columns]


def _positive_log_target(y):
    values = np.asarray(y, dtype=float)
    if np.any(values <= 0):
        raise ValueError("Target values must be positive before applying the log transform")
    return np.log(values)


def _exp_target(y):
    return np.exp(np.asarray(y, dtype=float))


def _perpetual_rejects_random_state(error: Exception) -> bool:
    message = str(error).lower()
    return "random_state" in message and (
        isinstance(error, TypeError)
        or "unknown keyword" in message
        or "unexpected keyword" in message
    )


def build_ridge_pipeline(
    selected: SelectedColumns,
    *,
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 1,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
) -> TransformedTargetRegressor:
    model = Pipeline(
        steps=[
            (
                "features",
                _build_text_feature_transformer(
                    selected,
                    tfidf_max_features=tfidf_max_features,
                    tfidf_min_df=tfidf_min_df,
                    tfidf_ngram_range=tfidf_ngram_range,
                ),
            ),
            ("ridge", Ridge(alpha=3.0)),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )


def build_elasticnet_pipeline(
    selected: SelectedColumns,
    alpha: float = 5e-05,
    l1_ratio: float = 0.655,
    max_iter: int = 20000,
    *,
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 1,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
) -> TransformedTargetRegressor:
    model = Pipeline(
        steps=[
            (
                "features",
                _build_text_feature_transformer(
                    selected,
                    tfidf_max_features=tfidf_max_features,
                    tfidf_min_df=tfidf_min_df,
                    tfidf_ngram_range=tfidf_ngram_range,
                ),
            ),
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


def build_xgboost_pipeline(
    selected: SelectedColumns,
    random_state: int = 42,
    device: str = "cpu",
) -> TransformedTargetRegressor:
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
                    **({"device": "cuda"} if device == "gpu" else {}),
                ),
            ),
        ]
    )
    return TransformedTargetRegressor(
        regressor=model,
        func=_positive_log_target,
        inverse_func=_exp_target,
    )


def build_perpetual_pipeline(selected: SelectedColumns, random_state: int = 42) -> TransformedTargetRegressor:
    try:
        from perpetual import PerpetualRegressor
    except ModuleNotFoundError as exc:
        raise ImportError("perpetual is not installed") from exc

    model_kwargs = {
        "objective": "SquaredLoss",
        "budget": 0.5,
    }
    try:
        regressor = PerpetualRegressor(**model_kwargs, random_state=random_state)
    except (TypeError, ValueError) as exc:
        if not _perpetual_rejects_random_state(exc):
            raise
        regressor = PerpetualRegressor(**model_kwargs)

    model = Pipeline(
        steps=[
            ("features", _build_feature_transformer(selected)),
            ("perpetual", regressor),
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
                FunctionTransformer(_select_columns, kw_args={"columns": selected.non_text_features}),
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
