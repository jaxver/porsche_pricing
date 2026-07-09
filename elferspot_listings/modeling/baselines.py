from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.base import BaseEstimator, RegressorMixin, clone
from sklearn.compose import ColumnTransformer, TransformedTargetRegressor
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.impute import SimpleImputer
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.model_selection import KFold
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


class StackedEnsembleRegressor(BaseEstimator, RegressorMixin):
    def __init__(
        self,
        selected: SelectedColumns,
        random_state: int = 42,
        n_splits: int = 5,
        meta_alpha: float = 1.0,
        prediction_floor: float = 1.0,
        tfidf_max_features: int = 5000,
        tfidf_min_df: int = 1,
        tfidf_ngram_range: tuple[int, int] = (1, 2),
    ):
        self.selected = selected
        self.random_state = random_state
        self.n_splits = n_splits
        self.meta_alpha = meta_alpha
        self.prediction_floor = prediction_floor
        self.tfidf_max_features = tfidf_max_features
        self.tfidf_min_df = tfidf_min_df
        self.tfidf_ngram_range = tfidf_ngram_range

    def _build_base_estimators(self):
        return [
            (
                "ridge",
                build_ridge_pipeline(
                    self.selected,
                    tfidf_max_features=self.tfidf_max_features,
                    tfidf_min_df=self.tfidf_min_df,
                    tfidf_ngram_range=self.tfidf_ngram_range,
                ),
            ),
            (
                "elasticnet",
                build_elasticnet_pipeline(
                    self.selected,
                    tfidf_max_features=self.tfidf_max_features,
                    tfidf_min_df=self.tfidf_min_df,
                    tfidf_ngram_range=self.tfidf_ngram_range,
                ),
            ),
        ]

    def _predict_fold(self, model: Any, X: Any) -> np.ndarray:
        predictions = np.asarray(model.predict(X), dtype=float).reshape(-1)
        predictions = np.where(np.isfinite(predictions), predictions, float(self.prediction_floor))
        return np.clip(predictions, float(self.prediction_floor), None)

    def fit(self, X: Any, y: Any):
        y_values = np.asarray(y, dtype=float).reshape(-1)
        if np.any(y_values <= 0):
            raise ValueError("Target values must be positive before training the stacked ensemble")

        base_estimators = self._build_base_estimators()
        n_samples = len(X)
        n_splits = min(int(self.n_splits), n_samples)
        if n_splits < 2:
            raise ValueError("StackedEnsembleRegressor requires at least 2 rows")

        splitter = KFold(n_splits=n_splits, shuffle=True, random_state=self.random_state)
        oof_predictions = np.zeros((n_samples, len(base_estimators)), dtype=float)

        for fold_train_indices, fold_valid_indices in splitter.split(X, y_values):
            X_fold_train = X.iloc[fold_train_indices] if hasattr(X, "iloc") else X[fold_train_indices]
            X_fold_valid = X.iloc[fold_valid_indices] if hasattr(X, "iloc") else X[fold_valid_indices]
            y_fold_train = y_values[fold_train_indices]

            for column_index, (_model_name, estimator) in enumerate(base_estimators):
                fold_model = clone(estimator)
                fold_model.fit(X_fold_train, y_fold_train)
                oof_predictions[fold_valid_indices, column_index] = self._predict_fold(fold_model, X_fold_valid)

        self.oof_predictions_ = np.clip(oof_predictions, float(self.prediction_floor), None)
        self.base_model_names_ = tuple(model_name for model_name, _estimator in base_estimators)
        self.base_models_ = {}
        for model_name, estimator in base_estimators:
            fitted_model = clone(estimator)
            fitted_model.fit(X, y_values)
            self.base_models_[model_name] = fitted_model

        self.meta_model_ = Ridge(alpha=float(self.meta_alpha))
        self.meta_model_.fit(np.log(self.oof_predictions_), np.log(y_values))
        self.ensemble_strategy_, self.average_model_names_ = self._select_ensemble_strategy(self.oof_predictions_, y_values)
        return self

    def _select_ensemble_strategy(self, base_predictions: np.ndarray, y_values: np.ndarray) -> tuple[str, tuple[str, ...]]:
        candidates: dict[tuple[str, tuple[str, ...]], np.ndarray] = {}
        ridge_elasticnet_names = tuple(name for name in ("ridge", "elasticnet") if name in self.base_model_names_)
        if len(ridge_elasticnet_names) >= 2:
            indices = [self.base_model_names_.index(name) for name in ridge_elasticnet_names]
            candidates[("geometric_mean", ridge_elasticnet_names)] = np.exp(np.log(base_predictions[:, indices]).mean(axis=1))
            return "geometric_mean", ridge_elasticnet_names

        for column_index, model_name in enumerate(self.base_model_names_):
            candidates[("single", (model_name,))] = base_predictions[:, column_index]

        best_key = min(candidates, key=lambda key: float(np.mean(np.abs(candidates[key] - y_values))))
        return best_key

    def predict(self, X: Any):
        check_is_fitted(self, ["base_models_", "meta_model_"])
        stacked_predictions = np.column_stack(
            [self._predict_fold(self.base_models_[model_name], X) for model_name in self.base_model_names_]
        )
        if self.ensemble_strategy_ == "geometric_mean":
            indices = [self.base_model_names_.index(name) for name in self.average_model_names_]
            return np.exp(np.log(stacked_predictions[:, indices]).mean(axis=1))
        if self.ensemble_strategy_ == "single":
            index = self.base_model_names_.index(self.average_model_names_[0])
            return stacked_predictions[:, index]

        meta_log_predictions = np.asarray(self.meta_model_.predict(np.log(stacked_predictions)), dtype=float).reshape(-1)
        meta_predictions = np.exp(meta_log_predictions)
        meta_predictions = np.where(np.isfinite(meta_predictions), meta_predictions, float(self.prediction_floor))
        return np.clip(meta_predictions, float(self.prediction_floor), None)


def build_stacked_ensemble_pipeline(
    selected: SelectedColumns,
    random_state: int = 42,
    n_splits: int = 5,
    meta_alpha: float = 1.0,
    prediction_floor: float = 1.0,
    tfidf_max_features: int = 5000,
    tfidf_min_df: int = 1,
    tfidf_ngram_range: tuple[int, int] = (1, 2),
) -> StackedEnsembleRegressor:
    return StackedEnsembleRegressor(
        selected=selected,
        random_state=random_state,
        n_splits=n_splits,
        meta_alpha=meta_alpha,
        prediction_floor=prediction_floor,
        tfidf_max_features=tfidf_max_features,
        tfidf_min_df=tfidf_min_df,
        tfidf_ngram_range=tfidf_ngram_range,
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
