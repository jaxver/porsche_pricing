from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.utils.validation import check_is_fitted

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
    "weissach_package",
    "pccb",
    "ceramic_brakes",
    "bucket_seats",
    "clubsport_package",
    "front_axle_lift",
    "sport_chrono",
    "manual_transmission_text",
    "paint_to_sample_text",
    "manthey",
    "ruf",
    "techart",
    "carbon_package",
    "lightweight_package",
    "full_leather",
    "carbon_bucket_seats",
}


class MedianRegressor(BaseEstimator, RegressorMixin):
    def fit(self, X: Any, y: Any):
        self.median_ = float(np.median(np.asarray(y, dtype=float)))
        return self

    def predict(self, X: Any):
        if not hasattr(self, "median_"):
            raise AttributeError("MedianRegressor is not fitted yet")
        return np.full(shape=len(X), fill_value=self.median_, dtype=float)


def _tabular_only_selected_columns(selected: SelectedColumns) -> SelectedColumns:
    return SelectedColumns(
        target=selected.target,
        numeric=selected.numeric,
        categorical=selected.categorical,
    )


class HighPriceSpecialistRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        selected: SelectedColumns,
        high_price_threshold: float = 250_000.0,
        routing_threshold: float = 0.5,
        min_specialist_rows: int = 4,
        random_state: int = 42,
    ):
        self.selected = selected
        self.high_price_threshold = high_price_threshold
        self.routing_threshold = routing_threshold
        self.min_specialist_rows = min_specialist_rows
        self.random_state = random_state

    def _build_general_regressor(self) -> TransformedTargetRegressor:
        selected = _tabular_only_selected_columns(self.selected)
        model = Pipeline(
            steps=[
                (
                    "features",
                    _build_feature_transformer(
                        selected,
                        numeric_with_mean=False,
                        categorical_sparse_output=True,
                        sparse_threshold=1.0,
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

    def _build_fallback_regressor(self) -> MedianRegressor:
        return MedianRegressor()

    def _build_classifier(self) -> Pipeline:
        selected = _tabular_only_selected_columns(self.selected)
        return Pipeline(
            steps=[
                (
                    "features",
                    _build_feature_transformer(
                        selected,
                        numeric_with_mean=False,
                        categorical_sparse_output=True,
                        sparse_threshold=1.0,
                    ),
                ),
                (
                    "logistic",
                    LogisticRegression(
                        class_weight="balanced",
                        max_iter=1000,
                        random_state=self.random_state,
                    ),
                ),
            ]
        )

    def fit(self, X: Any, y: Any):
        selected = _tabular_only_selected_columns(self.selected)
        if not selected.non_text_features:
            self.general_regressor_ = self._build_fallback_regressor()
            self.general_regressor_.fit(X, y)
            self.classifier_ = None
            self.specialist_regressor_ = None
            self.high_price_rows_ = int(np.count_nonzero(np.asarray(y, dtype=float).reshape(-1) >= float(self.high_price_threshold)))
            return self

        y_values = np.asarray(y, dtype=float).reshape(-1)
        self.general_regressor_ = self._build_general_regressor()
        self.general_regressor_.fit(X, y_values)

        high_mask = y_values >= float(self.high_price_threshold)
        self.high_price_rows_ = int(np.count_nonzero(high_mask))

        if np.unique(high_mask.astype(int)).size > 1:
            self.classifier_ = self._build_classifier()
            self.classifier_.fit(X, high_mask.astype(int))
        else:
            self.classifier_ = None
        self.all_training_rows_high_price_ = bool(high_mask.all())

        if self.high_price_rows_ >= int(self.min_specialist_rows):
            self.specialist_regressor_ = self._build_general_regressor()
            self.specialist_regressor_.fit(X.loc[high_mask], y_values[high_mask])
        else:
            self.specialist_regressor_ = None

        return self

    def predict(self, X: Any):
        check_is_fitted(self, ["general_regressor_"])
        general_predictions = np.asarray(self.general_regressor_.predict(X), dtype=float)
        if self.specialist_regressor_ is not None and getattr(self, "all_training_rows_high_price_", False):
            return np.asarray(self.specialist_regressor_.predict(X), dtype=float)
        if self.classifier_ is None or self.specialist_regressor_ is None:
            return general_predictions

        specialist_probabilities = np.asarray(self.classifier_.predict_proba(X)[:, 1], dtype=float)
        specialist_predictions = np.asarray(self.specialist_regressor_.predict(X), dtype=float)
        return np.where(specialist_probabilities >= float(self.routing_threshold), specialist_predictions, general_predictions)


def build_high_price_specialist_pipeline(
    selected: SelectedColumns,
    *,
    high_price_threshold: float = 250_000.0,
    routing_threshold: float = 0.5,
    min_specialist_rows: int = 4,
    random_state: int = 42,
) -> HighPriceSpecialistRegressor:
    return HighPriceSpecialistRegressor(
        selected=selected,
        high_price_threshold=high_price_threshold,
        routing_threshold=routing_threshold,
        min_specialist_rows=min_specialist_rows,
        random_state=random_state,
    )


def _build_feature_transformer(
    selected: SelectedColumns,
    *,
    numeric_with_mean: bool = True,
    categorical_sparse_output: bool = False,
    sparse_threshold: float = 0.3,
) -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler(with_mean=numeric_with_mean)),
        ]
    )
    categorical_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="constant", fill_value="missing")),
            ("encoder", OneHotEncoder(handle_unknown="ignore", sparse_output=categorical_sparse_output)),
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
        sparse_threshold=sparse_threshold,
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
