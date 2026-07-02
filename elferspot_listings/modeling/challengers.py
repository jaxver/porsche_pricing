from __future__ import annotations

import inspect
import time
import shutil
from pathlib import Path

import pandas as pd


_INSTALL_GUIDANCE = 'python -m pip install -e ".[advanced]"'
_TABPFN_CLIENT_INSTALL_GUIDANCE = 'python -m pip install -e ".[advanced]"'
_TABPFN_BROWSER_AUTH_GUIDANCE = (
    "TabPFN browser/license authentication failed. Accept the Prior Labs license in a browser manually, "
    "set `TABPFN_TOKEN` in the environment before rerunning, and avoid browser auth from proxied or "
    "non-interactive Windows runs."
)
_TABPFN_CLIENT_ACCESS_GUIDANCE = (
    "tabpfn-client API authentication/access failed. Set or access your Prior Labs access token before rerunning, "
    "and retry after resolving any tabpfn-client quota, network, or service availability issue."
)
_TABFM_LOAD_GUIDANCE = (
    "TabFM load/auth/download failed. Re-run after allowing network access, resolving any auth or license prompt, "
    "and remember the first run may download the published weights."
)
_TABPFN_CUDA_UNAVAILABLE_GUIDANCE = (
    "local TabPFN GPU requested but CUDA is unavailable or Torch is not compiled with CUDA; rerun with `--device cpu` "
    "or install a CUDA-enabled PyTorch build."
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


def _path_is_symlink_or_junction(path: Path) -> bool:
    is_junction = getattr(path, "is_junction", None)
    return bool(path.is_symlink() or (callable(is_junction) and is_junction()))


def _validate_autogluon_cleanup_target(output_dir_path: Path, output_path: Path) -> None:
    allowed_cleanup_dir = output_dir_path / "autogluon"
    try:
        output_dir_resolved = output_dir_path.resolve(strict=False)
        output_resolved = output_path.resolve(strict=False)
        allowed_resolved = allowed_cleanup_dir.resolve(strict=False)
    except OSError as exc:
        raise ValueError(
            "clean_output is only allowed for a dedicated AutoGluon artifact directory under the current output directory."
        ) from exc

    if _path_is_symlink_or_junction(output_dir_path) or _path_is_symlink_or_junction(output_path):
        raise ValueError(
            "clean_output is only allowed for a dedicated AutoGluon artifact directory under the current output directory."
        )
    if output_path.name != "autogluon" or output_resolved != allowed_resolved or not output_resolved.is_relative_to(output_dir_resolved):
        raise ValueError(
            "clean_output is only allowed for a dedicated AutoGluon artifact directory under the current output directory."
        )


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


def _is_tabpfn_cuda_unavailable_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    return any(
        marker in message
        for marker in (
            "torch not compiled with cuda enabled",
            "cuda is unavailable",
            "cuda unavailable",
            "cuda not available",
            "torch/cuda is not available",
        )
    )


def _is_tabpfn_client_access_failure(exc: BaseException) -> bool:
    message = str(exc).lower()
    plain_failure_markers = (
        "authentication required",
        "missing prior labs access token",
        "api access denied",
        "invalid token",
        "unauthorized",
    )
    auth_markers = (
        "unauthorized",
        "forbidden",
        "authentication",
        "auth",
        "login",
        "signin",
        "sign in",
        "prior labs",
        "access token",
        "token",
        "license",
    )
    access_markers = (
        "api access",
        "api",
        "quota",
        "rate limit",
        "too many requests",
        "connection",
        "network",
        "timeout",
        "dns",
        "socket",
        "ssl",
        "certificate",
        "proxy",
        "fetch",
        "client service",
        "connect",
        "502",
        "503",
        "504",
        "429",
        "401",
        "403",
    )
    service_failure_markers = (
        "503 service unavailable",
        "connection timeout",
        "quota exceeded",
        "dns lookup failed",
        "proxy connection failed",
        "rate limit exceeded",
        "service unavailable",
        "network unreachable",
    )
    return any(marker in message for marker in plain_failure_markers) or (
        any(marker in message for marker in auth_markers) and any(marker in message for marker in access_markers)
    ) or any(marker in message for marker in service_failure_markers)


def _is_tabfm_load_failure(exc: BaseException) -> bool:
    message = str(exc).lower().replace("_", " ")
    return any(
        marker in message
        for marker in (
            "authentication",
            "license",
            "access token",
            "browser auth",
            "prior labs",
            "unauthorized",
            "forbidden",
            "connection timeout",
            "connection refused",
            "service unavailable",
            "network error",
            "proxy connection failed",
            "dns lookup failed",
            "network unreachable",
            "ssl certificate",
            "certificate verify failed",
            "403",
            "429",
            "502",
            "503",
            "504",
        )
    )


def run_tabfm_regression(
    X_train,
    y_train,
    X_test,
    random_state: int = 42,
) -> tuple[object, pd.DataFrame, dict]:
    start = time.perf_counter()
    try:
        from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0
    except ImportError as exc:
        raise _optional_dependency_error("TabFM", exc) from exc

    try:
        checkpoint = tabfm_v1_0_0.load(model_type="regression")
    except Exception as exc:
        if _is_tabfm_load_failure(exc):
            raise _optional_dependency_error("TabFM", exc, _TABFM_LOAD_GUIDANCE) from exc
        raise

    model = TabFMRegressor(model=checkpoint)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    predicted_values = pd.Series(predictions, index=X_test.index, dtype=float)
    prediction_frame = pd.DataFrame({"predicted_price_eur": predicted_values.to_numpy(dtype=float)}, index=X_test.index)
    metadata = {
        "model_name": "tabfm",
        "backend": "pytorch",
        "runtime_seconds": time.perf_counter() - start,
        "notes": "Using the published TabFM PyTorch checkpoint. First run may download weights.",
    }
    return model, prediction_frame, metadata


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
        if device == "gpu" and _is_tabpfn_cuda_unavailable_failure(exc):
            raise _optional_dependency_error("TabPFN", exc, _TABPFN_CUDA_UNAVAILABLE_GUIDANCE) from exc
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
        import tabpfn_client
    except ImportError as exc:
        raise _optional_dependency_error("tabpfn-client", exc, f"Install tabpfn-client with `{_TABPFN_CLIENT_INSTALL_GUIDANCE}`.") from exc

    tabpfn_init = getattr(tabpfn_client, "init", None)
    if callable(tabpfn_init):
        try:
            tabpfn_init()
        except Exception as exc:
            if _is_tabpfn_client_access_failure(exc):
                raise _optional_dependency_error("tabpfn-client", exc, _TABPFN_CLIENT_ACCESS_GUIDANCE) from exc
            raise

    try:
        TabPFNRegressor = tabpfn_client.TabPFNRegressor
    except AttributeError as exc:
        raise _optional_dependency_error("tabpfn-client", exc, _TABPFN_CLIENT_ACCESS_GUIDANCE) from exc

    model_kwargs: dict[str, object] = {
        "random_state": random_state,
        "thinking_mode": thinking_mode,
        "thinking_effort": thinking_effort,
        "thinking_metric": thinking_metric,
    }
    if thinking_timeout_s is not None:
        model_kwargs["thinking_timeout_s"] = thinking_timeout_s

    try:
        model = TabPFNRegressor(**model_kwargs)
        model.fit(X_train, y_train)
        predictions = model.predict(X_test)
    except Exception as exc:
        if _is_tabpfn_client_access_failure(exc):
            raise _optional_dependency_error("tabpfn-client", exc, _TABPFN_CLIENT_ACCESS_GUIDANCE) from exc
        raise

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
    presets: str = "best_quality",
    dynamic_stacking: bool | None = None,
    clean_output: bool = False,
) -> tuple[object, object, pd.DataFrame, dict]:
    start = time.perf_counter()
    try:
        from autogluon.tabular import TabularPredictor
    except ImportError as exc:
        raise _optional_dependency_error("AutoGluon", exc) from exc

    if dynamic_stacking is not None and type(dynamic_stacking) is not bool:
        raise TypeError("dynamic_stacking must be None, True, or False")

    output_dir_path = Path(output_dir)
    output_path = Path(artifact_dir) if artifact_dir is not None else output_dir_path / "autogluon"
    if clean_output:
        _validate_autogluon_cleanup_target(output_dir_path, output_path)
        if output_path.exists():
            if output_path.is_dir():
                shutil.rmtree(output_path)
            else:
                output_path.unlink()
    output_path.mkdir(parents=True, exist_ok=True)

    fit_kwargs: dict[str, object] = {
        "time_limit": time_limit,
        "presets": presets,
    }
    if dynamic_stacking is not None:
        fit_kwargs["dynamic_stacking"] = dynamic_stacking

    predictor = TabularPredictor(label=target, path=str(output_path), problem_type="regression").fit(train_df, **fit_kwargs)
    features = test_df.drop(columns=[target], errors="ignore")
    predictions = predictor.predict(features)
    leaderboard = predictor.leaderboard(test_df, silent=True)
    leaderboard.to_csv(output_path / "leaderboard.csv", index=False)
    metadata = {
        "model_name": "autogluon",
        "runtime_seconds": time.perf_counter() - start,
        "time_limit_seconds": time_limit,
        "presets": presets,
        "dynamic_stacking": dynamic_stacking,
        "clean_output": clean_output,
    }
    return predictor, predictions, leaderboard, metadata
