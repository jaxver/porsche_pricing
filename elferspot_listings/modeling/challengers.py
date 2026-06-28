from __future__ import annotations

import inspect
import time
from pathlib import Path

import pandas as pd


_INSTALL_GUIDANCE = "python -m pip install -r requirements-advanced.txt"
_TABPFN_CLIENT_INSTALL_GUIDANCE = "python -m pip install -r requirements-advanced.txt"
_TABPFN_BROWSER_AUTH_GUIDANCE = (
    "TabPFN browser/license authentication failed. Accept the Prior Labs license in a browser manually, "
    "set `TABPFN_TOKEN` in the environment before rerunning, and avoid browser auth from proxied or "
    "non-interactive Windows runs."
)


class OptionalDependencyNotInstalledError(ImportError, RuntimeError):
    def __init__(self, package_name: str, message: str | None = None):
        super().__init__(message or f"Install {package_name} with `{_INSTALL_GUIDANCE}`.")
        self.package_name = package_name


def _optional_dependency_error(
    package_name: str,
    exc: BaseException,
    message: str | None = None,
) -> OptionalDependencyNotInstalledError:
    return OptionalDependencyNotInstalledError(package_name, message)


def _is_tabpfn_browser_auth_failure(exc: BaseException) -> bool:
    if isinstance(exc, OSError) and getattr(exc, "winerror", None) == 10038:
        return True

    message = str(exc).lower()
    return (
        "winerror 10038" in message
        or "tabpfn_token" in message
        or "prior labs" in message
        or "browser auth" in message
        or ("tabpfn" in message and "license" in message)
        or ("tabpfn" in message and "login" in message)
        or ("tabpfn" in message and "signin" in message)
        or "select.select([sys.stdin]" in message
    )


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

    try:
        model = TabPFNRegressor(**model_kwargs)
        model.fit(X_train, y_train)
    except Exception as exc:
        if _is_tabpfn_browser_auth_failure(exc):
            raise _optional_dependency_error("TabPFN", exc, _TABPFN_BROWSER_AUTH_GUIDANCE) from exc
        raise
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


def run_tabpfn_client_regression(
    X_train,
    y_train,
    X_test,
    random_state: int = 42,
    thinking_mode: bool = False,
    thinking_effort: str = "medium",
    thinking_metric: str = "rmse",
    thinking_timeout_s: float | int | None = None,
) -> tuple[object, object, dict]:
    start = time.perf_counter()
    try:
        from tabpfn_client import TabPFNRegressor
    except ImportError as exc:
        raise _optional_dependency_error("tabpfn-client", exc, f"Install tabpfn-client with `{_TABPFN_CLIENT_INSTALL_GUIDANCE}`.") from exc

    model_kwargs: dict[str, object] = {
        "random_state": random_state,
        "thinking_mode": thinking_mode,
        "thinking_effort": thinking_effort,
        "thinking_metric": thinking_metric,
    }
    if thinking_timeout_s is not None:
        model_kwargs["thinking_timeout_s"] = thinking_timeout_s

    model = TabPFNRegressor(**model_kwargs)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    model_name = "tabpfn_client_thinking" if thinking_mode else "tabpfn_client"
    notes = ["backend=client"]
    if thinking_mode:
        notes.append(f"thinking_mode=True effort={thinking_effort} metric={thinking_metric}")
        if thinking_timeout_s is not None:
            notes.append(f"timeout={thinking_timeout_s}")
    metadata = {
        "model_name": model_name,
        "backend": "client",
        "runtime_seconds": time.perf_counter() - start,
        "notes": " ".join(notes),
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
