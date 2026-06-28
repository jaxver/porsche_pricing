from __future__ import annotations

import inspect
import time
from pathlib import Path

import pandas as pd


_INSTALL_GUIDANCE = "python -m pip install -r requirements-advanced.txt"


class OptionalDependencyNotInstalledError(ImportError, RuntimeError):
    def __init__(self, package_name: str):
        super().__init__(f"Install {package_name} with `{_INSTALL_GUIDANCE}`.")


def _optional_dependency_error(package_name: str, exc: ImportError) -> OptionalDependencyNotInstalledError:
    return OptionalDependencyNotInstalledError(package_name)


def run_tabpfn_regression(
    X_train,
    y_train,
    X_test,
    random_state: int = 42,
    model_path: str | None = None,
    model_name: str = "tabpfn_default",
    device: str = "cpu",
    gpu_devices: str | None = None,
) -> tuple[object, object, dict]:
    start = time.perf_counter()
    if model_path not in (None, "default") and not model_path.endswith(".ckpt"):
        raise ValueError(
            f"Unsupported TabPFN model_path '{model_path}'. Valid direct values are None, 'default', or a string ending in .ckpt."
        )
    try:
        from tabpfn import TabPFNRegressor
    except ImportError as exc:
        raise _optional_dependency_error("TabPFN", exc) from exc

    checkpoint_label = Path(model_path).name if model_path not in (None, "default") else None
    model_kwargs: dict[str, object] = {"random_state": random_state}
    if model_path not in (None, "default"):
        model_kwargs["model_path"] = model_path

    device_note = None
    if device == "gpu":
        accepts_device = False
        try:
            accepts_device = "device" in inspect.signature(TabPFNRegressor.__init__).parameters
        except (TypeError, ValueError):
            accepts_device = False
        if accepts_device:
            model_kwargs["device"] = "cuda"
        else:
            device_note = "Requested GPU device, but TabPFN constructor did not accept a device parameter."

    model = TabPFNRegressor(**model_kwargs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)
    notes = "Default TabPFN checkpoint."
    if checkpoint_label is not None:
        notes = f"Using TabPFN checkpoint {checkpoint_label}."
    if device_note is not None:
        notes = f"{notes} {device_note}"
    metadata = {
        "model_name": model_name,
        "model_path": checkpoint_label,
        "runtime_seconds": time.perf_counter() - start,
        "notes": f"{notes} First run may download TabPFN checkpoints.",
    }
    return model, predictions, metadata


def run_autogluon_regression(
    train_df: pd.DataFrame,
    test_df: pd.DataFrame,
    target: str,
    output_dir: str | Path,
    time_limit: int = 600,
    artifact_dir: str | Path | None = None,
) -> tuple[object, object, pd.DataFrame, dict]:
    start = time.perf_counter()
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise _optional_dependency_error("AutoGluon", exc) from exc

    output_path = Path(artifact_dir) if artifact_dir is not None else Path(output_dir) / "autogluon"
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
