from __future__ import annotations

import time
from pathlib import Path

import pandas as pd


_INSTALL_GUIDANCE = "python -m pip install -r requirements-advanced.txt"


def _optional_dependency_error(package_name: str, exc: ImportError) -> RuntimeError:
    return RuntimeError(f"Install {package_name} with `{_INSTALL_GUIDANCE}`.")


def run_tabpfn_regression(X_train, y_train, X_test, random_state: int = 42) -> tuple[object, object, dict]:
    start = time.perf_counter()
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise _optional_dependency_error("TabPFN", exc) from exc

    model = TabPFNRegressor(random_state=random_state)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    metadata = {
        "model_name": "tabpfn",
        "runtime_seconds": time.perf_counter() - start,
        "notes": "First run may download TabPFN checkpoints.",
    }
    return model, predictions, metadata


def run_autogluon_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    output_dir: str | Path,
    time_limit: int = 600,
) -> tuple[object, object, pd.DataFrame, dict]:
    start = time.perf_counter()
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise _optional_dependency_error("AutoGluon", exc) from exc

    output_path = Path(output_dir) / "autogluon"
    output_path.mkdir(parents=True, exist_ok=True)

    predictor = TabularPredictor(label=target, path=str(output_path), problem_type="regression").fit(
        train_df,
        time_limit=time_limit,
        presets="best_quality",
    )
    features = test_df.drop(columns=[target], errors="ignore")
    predictions = predictor.predict(features)
    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)
    metadata = {
        "model_name": "autogluon",
        "runtime_seconds": time.perf_counter() - start,
        "time_limit_seconds": time_limit,
        "presets": "best_quality",
    }
    return predictor, predictions, leaderboard, metadata
