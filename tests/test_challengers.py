from __future__ import annotations

import importlib
import importlib.util
import json
import os
import sys
import types
import tempfile
from pathlib import Path
from typing import cast

import pandas as pd
import pytest


def _tabfm_regression_config() -> dict[str, object]:
    return {
        "col_nhead": 4,
        "col_num_blocks": 3,
        "col_num_inds": 256,
        "decoder_hidden": None,
        "embed_dim": 256,
        "feature_group_size": 3,
        "ff_factor": 4,
        "icl_nhead": 8,
        "icl_num_blocks": 24,
        "is_classifier": False,
        "max_classes": 10,
        "num_freq": 32,
        "row_nhead": 8,
        "row_num_blocks": 3,
        "row_num_cls": 8,
    }


def _write_tabfm_snapshot(snapshot_dir: Path, include_checkpoint: bool = True) -> Path:
    regression_dir = snapshot_dir / "regression"
    regression_dir.mkdir(parents=True, exist_ok=True)
    (regression_dir / "config.json").write_text(json.dumps(_tabfm_regression_config()), encoding="utf-8")
    if include_checkpoint:
        (regression_dir / "model.safetensors").write_bytes(b"placeholder")
    return snapshot_dir


def _install_tabfm_test_doubles(
    monkeypatch,
    *,
    snapshot_dir: Path | None = None,
    snapshot_error: Exception | None = None,
    load_file_error: Exception | None = None,
    to_error: Exception | None = None,
    fit_error: Exception | None = None,
) -> dict[str, object]:
    import tabfm  # type: ignore[import-not-found]
    import tabfm.src.pytorch.model as tabfm_model  # type: ignore[import-not-found]

    captured: dict[str, object] = {}

    class FakeTabFM:
        def __init__(self, **config):
            captured["config"] = config

        def load_state_dict(self, state_dict, strict=True):
            captured["state_dict"] = state_dict
            captured["strict"] = strict
            return self

        def to(self, device):
            captured["device"] = device
            if to_error is not None:
                raise to_error
            return self

        def eval(self):
            captured["eval"] = True
            return self

    class FakeTabFMRegressor:
        def __init__(self, model):
            captured["model"] = model

        def fit(self, X_train, y_train):
            if fit_error is not None:
                raise fit_error
            captured["fit_shape"] = (len(X_train), len(y_train))
            return self

        def predict(self, X_test):
            captured["predict_index"] = list(X_test.index)
            return pd.Series([123.0] * len(X_test), index=X_test.index)

    def fake_snapshot_download(repo_id):
        captured["repo_id"] = repo_id
        captured["xet_env"] = os.environ.get("HF_HUB_DISABLE_XET")
        if snapshot_error is not None:
            raise snapshot_error
        if snapshot_dir is None:
            raise AssertionError("snapshot_dir is required when snapshot_error is not set")
        return str(snapshot_dir)

    def fake_load_file(path, device=None):
        captured["checkpoint_path"] = Path(path)
        captured["load_device"] = device
        if load_file_error is not None:
            raise load_file_error
        return {"fake_weight": 1.0}

    monkeypatch.setattr(tabfm_model, "TabFM", FakeTabFM)
    monkeypatch.setattr(tabfm, "TabFMRegressor", FakeTabFMRegressor)
    monkeypatch.setattr("huggingface_hub.snapshot_download", fake_snapshot_download)
    monkeypatch.setattr("safetensors.torch.load_file", fake_load_file)
    return captured


def test_challengers_module_imports_without_advanced_dependencies():
    module = importlib.import_module("elferspot_listings.modeling.challengers")

    assert hasattr(module, "run_tabpfn_regression")
    assert hasattr(module, "run_tabfm_regression")
    assert hasattr(module, "run_autogluon_regression")


def test_optional_dependency_error_subclasses_importerror_and_runtimeerror():
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError

    assert issubclass(OptionalDependencyNotInstalledError, ImportError)
    assert issubclass(OptionalDependencyNotInstalledError, RuntimeError)


def test_run_tabpfn_regression_raises_helpful_error_when_dependency_is_missing():
    if importlib.util.find_spec("tabpfn") is not None:
        pytest.skip("tabpfn is installed in this environment")

    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(RuntimeError, match=r"\.\[advanced\]"):
        run_tabpfn_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_loads_safetensors_checkpoint_and_moves_to_gpu(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabfm_regression

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = _write_tabfm_snapshot(Path(tmpdir))
        captured = _install_tabfm_test_doubles(monkeypatch, snapshot_dir=snapshot_dir)
        monkeypatch.delenv("HF_HUB_DISABLE_XET", raising=False)

        X_train = pd.DataFrame({"feature": [1.0, 2.0]})
        y_train = pd.Series([10.0, 20.0])
        X_test = pd.DataFrame({"feature": [3.0]})

        model, predictions, metadata = run_tabfm_regression(X_train, y_train, X_test, device="gpu")

    assert captured["repo_id"] == "google/tabfm-1.0.0-pytorch"
    assert captured["xet_env"] == "1"
    assert str(captured["checkpoint_path"]).endswith("model.safetensors")
    assert captured["load_device"] == "cpu"
    assert cast(dict, captured["config"])["is_classifier"] is False
    assert captured["state_dict"] == {"fake_weight": 1.0}
    assert captured["device"] == "cuda"
    assert captured["eval"] is True
    assert captured["fit_shape"] == (2, 2)
    assert captured["predict_index"] == [0]
    assert os.environ.get("HF_HUB_DISABLE_XET") is None
    assert list(predictions["predicted_price_eur"]) == [123.0]
    assert metadata["model_name"] == "tabfm"
    assert metadata["backend"] == "pytorch"
    assert metadata["runtime_seconds"] >= 0
    assert "safetensors" in metadata["notes"].lower()


def test_run_tabfm_regression_raises_helpful_error_when_dependency_is_missing():
    try:
        tabfm_spec = importlib.util.find_spec("tabfm")
    except ModuleNotFoundError:
        tabfm_spec = None

    if tabfm_spec is not None:
        pytest.skip("tabfm is installed in this environment")

    from elferspot_listings.modeling.challengers import run_tabfm_regression

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(RuntimeError, match=r"\.\[advanced\]"):
        run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_auth_like_load_failures_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id: (_ for _ in ()).throw(RuntimeError("401 Unauthorized: Prior Labs license required")))

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"TabFM load/auth/download failed|Prior Labs|license"):
        run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_network_load_failures_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id: (_ for _ in ()).throw(RuntimeError("503 Service Unavailable: connection timeout while fetching weights")))

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"TabFM load/auth/download failed|connection timeout|503"):
        run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_certificate_load_failures_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    monkeypatch.setattr("huggingface_hub.snapshot_download", lambda repo_id: (_ for _ in ()).throw(RuntimeError("CERTIFICATE_VERIFY_FAILED while fetching checkpoint")))

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"TabFM load/auth/download failed|CERTIFICATE_VERIFY_FAILED|certificate verify failed"):
        run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_missing_checkpoint_layout_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = _write_tabfm_snapshot(Path(tmpdir), include_checkpoint=False)
        _install_tabfm_test_doubles(monkeypatch, snapshot_dir=snapshot_dir)

        X_train = pd.DataFrame({"feature": [1.0, 2.0]})
        y_train = pd.Series([10.0, 20.0])
        X_test = pd.DataFrame({"feature": [3.0]})

        with pytest.raises(OptionalDependencyNotInstalledError, match=r"TabFM load/auth/download failed|weights not found"):
            run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_paging_file_load_failures_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = _write_tabfm_snapshot(Path(tmpdir))
        paging_file_error = OSError("The paging file is too small for this operation to complete.")
        paging_file_error.winerror = 1455
        _install_tabfm_test_doubles(monkeypatch, snapshot_dir=snapshot_dir, load_file_error=paging_file_error)

        X_train = pd.DataFrame({"feature": [1.0, 2.0]})
        y_train = pd.Series([10.0, 20.0])
        X_test = pd.DataFrame({"feature": [3.0]})

        with pytest.raises(OptionalDependencyNotInstalledError, match=r"TabFM load/auth/download failed|paging file is too small"):
            run_tabfm_regression(X_train, y_train, X_test)


def test_run_tabfm_regression_converts_cuda_unavailable_errors_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabfm_regression

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = _write_tabfm_snapshot(Path(tmpdir))
        _install_tabfm_test_doubles(
            monkeypatch,
            snapshot_dir=snapshot_dir,
            to_error=AssertionError("Torch not compiled with CUDA enabled"),
        )

        X_train = pd.DataFrame({"feature": [1.0, 2.0]})
        y_train = pd.Series([10.0, 20.0])
        X_test = pd.DataFrame({"feature": [3.0]})

        with pytest.raises(OptionalDependencyNotInstalledError, match=r"CUDA|cuda|Torch not compiled with CUDA enabled|device cpu"):
            run_tabfm_regression(X_train, y_train, X_test, device="gpu")


def test_run_tabfm_regression_propagates_unrelated_training_errors(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabfm_regression

    with tempfile.TemporaryDirectory() as tmpdir:
        snapshot_dir = _write_tabfm_snapshot(Path(tmpdir))
        _install_tabfm_test_doubles(monkeypatch, snapshot_dir=snapshot_dir, fit_error=ValueError("bad data"))

        X_train = pd.DataFrame({"feature": [1.0, 2.0]})
        y_train = pd.Series([10.0, 20.0])
        X_test = pd.DataFrame({"feature": [3.0]})

        with pytest.raises(ValueError, match="bad data"):
            run_tabfm_regression(X_train, y_train, X_test)


def test_run_autogluon_regression_raises_helpful_error_when_dependency_is_missing():
    try:
        autogluon_spec = importlib.util.find_spec("autogluon.tabular")
    except ModuleNotFoundError:
        autogluon_spec = None

    if autogluon_spec is not None:
        pytest.skip("autogluon.tabular is installed in this environment")

    from elferspot_listings.modeling.challengers import run_autogluon_regression

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with pytest.raises(RuntimeError, match=r"\.\[advanced\]"):
        run_autogluon_regression(train_df, test_df, "price_in_eur", ".")


def test_run_autogluon_regression_passes_presets_and_dynamic_stacking_when_requested(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    captured = {}

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets, dynamic_stacking=None):
            captured["fit_kwargs"] = {
                "time_limit": time_limit,
                "presets": presets,
                "dynamic_stacking": dynamic_stacking,
            }
            return self

        def predict(self, features):
            return pd.Series([654.0], index=features.index)

        def leaderboard(self, data, silent):
            return pd.DataFrame({"model": ["fake"], "score": [0.0]})

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    _, _, _, metadata = run_autogluon_regression(
        train_df,
        test_df,
        "price_in_eur",
        ".",
        time_limit=15,
        presets="medium_quality",
        dynamic_stacking=True,
    )

    assert captured["fit_kwargs"] == {"time_limit": 15, "presets": "medium_quality", "dynamic_stacking": True}
    assert metadata["presets"] == "medium_quality"
    assert metadata["dynamic_stacking"] is True


def test_run_autogluon_regression_omits_dynamic_stacking_when_auto(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    captured = {}

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets):
            captured["fit_kwargs"] = {"time_limit": time_limit, "presets": presets}
            return self

        def predict(self, features):
            return pd.Series([654.0], index=features.index)

        def leaderboard(self, data, silent):
            return pd.DataFrame({"model": ["fake"], "score": [0.0]})

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    _, _, _, metadata = run_autogluon_regression(
        train_df,
        test_df,
        "price_in_eur",
        ".",
        time_limit=15,
        presets="high_quality",
    )

    assert captured["fit_kwargs"] == {"time_limit": 15, "presets": "high_quality"}
    assert metadata["presets"] == "high_quality"
    assert metadata["dynamic_stacking"] is None


def test_run_autogluon_regression_rejects_clean_output_for_missing_custom_artifact_dir(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets):
            raise AssertionError("fit should not run when clean_output is unsafe")

        def predict(self, features):
            raise AssertionError("predict should not run")

        def leaderboard(self, data, silent):
            raise AssertionError("leaderboard should not run")

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        output_dir = temp_path / "results"
        custom_artifact_dir = temp_path / "custom-artifact-dir"
        broken_target = temp_path / "missing-target"

        try:
            custom_artifact_dir.symlink_to(broken_target, target_is_directory=True)
        except OSError as exc:
            pytest.skip(f"symlink creation is unavailable in this environment: {exc}")

        with pytest.raises(ValueError, match="dedicated AutoGluon"):
            run_autogluon_regression(
                train_df,
                test_df,
                "price_in_eur",
                output_dir,
                time_limit=15,
                artifact_dir=custom_artifact_dir,
                clean_output=True,
            )

        assert not output_dir.exists()
        assert custom_artifact_dir.is_symlink()
        assert not custom_artifact_dir.exists()


def test_run_autogluon_regression_rejects_invalid_dynamic_stacking_value(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets, dynamic_stacking=None):
            raise AssertionError("fit should not run when dynamic_stacking is invalid")

        def predict(self, features):
            raise AssertionError("predict should not run")

        def leaderboard(self, data, silent):
            raise AssertionError("leaderboard should not run")

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with pytest.raises((TypeError, ValueError), match="dynamic_stacking"):
        run_autogluon_regression(
            train_df,
            test_df,
            "price_in_eur",
            ".",
            dynamic_stacking="auto",  # type: ignore[arg-type]
        )


def test_run_autogluon_regression_rejects_unsafe_default_cleanup_path(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets):
            raise AssertionError("fit should not run when cleanup path is unsafe")

        def predict(self, features):
            raise AssertionError("predict should not run")

        def leaderboard(self, data, silent):
            raise AssertionError("leaderboard should not run")

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        output_dir = temp_path / "results"
        artifact_dir = output_dir / "autogluon"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "stale.txt").write_text("old data", encoding="utf-8")

        monkeypatch.setattr(Path, "is_symlink", lambda self: self == artifact_dir)

        with pytest.raises(ValueError, match="dedicated AutoGluon"):
            run_autogluon_regression(
                train_df,
                test_df,
                "price_in_eur",
                output_dir,
                time_limit=15,
                clean_output=True,
            )

        assert artifact_dir.exists()
        assert (artifact_dir / "stale.txt").exists()


def test_run_autogluon_regression_cleans_existing_artifact_dir_when_requested(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    captured = {}

    class FakeTabularPredictor:
        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type

        def fit(self, train_df, time_limit, presets):
            captured["stale_exists_during_fit"] = Path(self.path, "stale.txt").exists()
            return self

        def predict(self, features):
            return pd.Series([654.0], index=features.index)

        def leaderboard(self, data, silent):
            return pd.DataFrame({"model": ["fake"], "score": [0.0]})

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with tempfile.TemporaryDirectory() as tmpdir:
        output_dir = Path(tmpdir) / "results"
        artifact_dir = output_dir / "autogluon"
        artifact_dir.mkdir(parents=True)
        (artifact_dir / "stale.txt").write_text("old data", encoding="utf-8")

        run_autogluon_regression(
            train_df,
            test_df,
            "price_in_eur",
            output_dir,
            time_limit=15,
            clean_output=True,
        )

        assert not (artifact_dir / "stale.txt").exists()
        assert captured["stale_exists_during_fit"] is False


def test_run_tabpfn_regression_uses_fake_module_and_returns_metadata(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            self.random_state = random_state
            self.model_path = model_path
            self.fit_calls = []
            self.predict_calls = []

        def fit(self, X_train, y_train):
            self.fit_calls.append((X_train.copy(), y_train.copy()))
            return self

        def predict(self, X_test):
            self.predict_calls.append(X_test.copy())
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    model, predictions, metadata = run_tabpfn_regression(
        X_train,
        y_train,
        X_test,
        random_state=17,
        model_path="nested/local/some.ckpt",
        model_name="tabpfn_custom",
    )

    assert isinstance(model, FakeTabPFNRegressor)
    assert model.random_state == 17
    assert model.model_path == "nested/local/some.ckpt"
    assert list(predictions) == [321.0]
    assert metadata["model_name"] == "tabpfn_custom"
    assert metadata["model_path"] == "some.ckpt"
    assert metadata["runtime_seconds"] >= 0
    assert "checkpoint" in metadata["notes"].lower()
    assert "nested/local/some.ckpt" not in metadata["notes"]
    assert len(model.fit_calls) == 1
    assert len(model.predict_calls) == 1


def test_run_tabpfn_client_regression_uses_fake_module_and_thinking_kwargs(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_client_regression

    captured = {}
    init_calls = []

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            captured["kwargs"] = kwargs
            self.fit_calls = []
            self.predict_calls = []

        def fit(self, X_train, y_train):
            self.fit_calls.append((X_train.copy(), y_train.copy()))
            return self

        def predict(self, X_test):
            self.predict_calls.append(X_test.copy())
            return pd.Series([456.0], index=X_test.index)

    def fake_init(*args, **kwargs):
        init_calls.append((args, kwargs))

    fake_client = types.ModuleType("tabpfn_client")
    fake_client.TabPFNRegressor = FakeTabPFNRegressor
    fake_client.init = fake_init
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake_client)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    model, predictions, metadata = run_tabpfn_client_regression(
        X_train,
        y_train,
        X_test,
        random_state=17,
        thinking_mode=True,
        thinking_effort="high",
        thinking_metric="mae",
        thinking_timeout_s=12,
    )

    assert isinstance(model, FakeTabPFNRegressor)
    assert init_calls == [((), {})]
    assert captured["kwargs"] == {
        "random_state": 17,
        "thinking_mode": True,
        "thinking_effort": "high",
        "thinking_metric": "mae",
        "thinking_timeout_s": 12,
    }
    assert list(predictions) == [456.0]
    assert metadata["model_name"] == "tabpfn_client_thinking"
    assert metadata["backend"] == "client"
    assert metadata["runtime_seconds"] >= 0
    assert "effort=high" in metadata["notes"]
    assert "metric=mae" in metadata["notes"]
    assert "timeout=12" in metadata["notes"]


def test_run_tabpfn_client_regression_converts_auth_and_api_failures_to_optional_dependency_error(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabpfn_client_regression

    def fake_init():
        raise RuntimeError("401 Unauthorized: missing Prior Labs access token for tabpfn-client API access")

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            raise AssertionError("constructor should not run after init failure")

    fake_client = types.ModuleType("tabpfn_client")
    fake_client.TabPFNRegressor = FakeTabPFNRegressor
    fake_client.init = fake_init
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake_client)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"tabpfn-client|Prior Labs|access token|API access") as excinfo:
        run_tabpfn_client_regression(X_train, y_train, X_test)

    assert "TABPFN_TOKEN" not in str(excinfo.value)


@pytest.mark.parametrize(
    "message",
    [
        "authentication required",
        "missing Prior Labs access token",
        "API access denied",
        "invalid token",
        "unauthorized",
    ],
)
def test_run_tabpfn_client_regression_converts_plain_auth_and_access_failures_to_optional_dependency_error(monkeypatch, message):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabpfn_client_regression

    def fake_init():
        raise RuntimeError(message)

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            raise AssertionError("constructor should not run after init failure")

    fake_client = types.ModuleType("tabpfn_client")
    fake_client.TabPFNRegressor = FakeTabPFNRegressor
    fake_client.init = fake_init
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake_client)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"tabpfn-client|Prior Labs|access token|API access"):
        run_tabpfn_client_regression(X_train, y_train, X_test)


@pytest.mark.parametrize(
    "message",
    [
        "503 Service Unavailable",
        "connection timeout",
        "quota exceeded",
        "DNS lookup failed",
        "proxy connection failed",
        "rate limit exceeded",
        "service unavailable",
        "network unreachable",
    ],
)
def test_run_tabpfn_client_regression_converts_network_quota_service_failures_to_optional_dependency_error(monkeypatch, message):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabpfn_client_regression

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            self.fit_calls = []

        def fit(self, X_train, y_train):
            raise RuntimeError(message)

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_client = types.ModuleType("tabpfn_client")
    fake_client.TabPFNRegressor = FakeTabPFNRegressor
    fake_client.init = lambda: None
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake_client)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"tabpfn-client|Prior Labs|access token|API access"):
        run_tabpfn_client_regression(X_train, y_train, X_test)


def test_run_tabpfn_client_regression_propagates_unrelated_training_errors(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_client_regression

    class FakeTabPFNRegressor:
        def __init__(self, **kwargs):
            pass

        def fit(self, X_train, y_train):
            raise ValueError("bad data")

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_client = types.ModuleType("tabpfn_client")
    fake_client.TabPFNRegressor = FakeTabPFNRegressor
    fake_client.init = lambda: None
    monkeypatch.setitem(sys.modules, "tabpfn_client", fake_client)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(ValueError, match="bad data"):
        run_tabpfn_client_regression(X_train, y_train, X_test)


def test_run_tabpfn_client_regression_raises_helpful_error_when_dependency_is_missing():
    if importlib.util.find_spec("tabpfn_client") is not None:
        pytest.skip("tabpfn_client is installed in this environment")

    from elferspot_listings.modeling.challengers import run_tabpfn_client_regression

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(RuntimeError, match=r"\.\[advanced\]"):
        run_tabpfn_client_regression(X_train, y_train, X_test)


def test_run_tabpfn_regression_raises_helpful_error_for_browser_auth_windows_socket_failure(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            self.random_state = random_state
            self.model_path = model_path

        def fit(self, X_train, y_train):
            raise OSError("[WinError 10038] An operation was attempted on something that is not a socket")

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(OptionalDependencyNotInstalledError, match=r"TABPFN_TOKEN|license|browser|proxied|non-interactive"):
        run_tabpfn_regression(X_train, y_train, X_test)


def test_run_tabpfn_regression_propagates_unrelated_training_errors(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            self.random_state = random_state
            self.model_path = model_path

        def fit(self, X_train, y_train):
            raise ValueError("bad data")

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(ValueError, match="bad data"):
        run_tabpfn_regression(X_train, y_train, X_test)


def test_run_tabpfn_regression_raises_helpful_error_when_local_gpu_cuda_is_unavailable(monkeypatch):
    from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError, run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None, device=None):
            self.random_state = random_state
            self.model_path = model_path
            self.device = device

        def fit(self, X_train, y_train):
            raise AssertionError("Torch not compiled with CUDA enabled")

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(
        OptionalDependencyNotInstalledError,
        match=r"CUDA|cuda|Torch not compiled with CUDA enabled|CPU|--device cpu",
    ):
        run_tabpfn_regression(X_train, y_train, X_test, device="gpu")


def test_run_tabpfn_regression_propagates_unrelated_assertion_errors(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None, device=None):
            self.random_state = random_state
            self.model_path = model_path
            self.device = device

        def fit(self, X_train, y_train):
            raise AssertionError("bad model invariant")

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(AssertionError, match="bad model invariant"):
        run_tabpfn_regression(X_train, y_train, X_test, device="gpu")


def test_run_tabpfn_regression_passes_gpu_device_when_constructor_accepts_it(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None, device=None):
            self.random_state = random_state
            self.model_path = model_path
            self.device = device

        def fit(self, X_train, y_train):
            return self

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    model, predictions, metadata = run_tabpfn_regression(X_train, y_train, X_test, device="gpu")

    assert model.device == "cuda"
    assert list(predictions) == [321.0]
    assert "not accepted" not in metadata["notes"].lower()


def test_run_tabpfn_regression_skips_gpu_device_when_constructor_rejects_it(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    captured = {}

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            captured["random_state"] = random_state
            captured["model_path"] = model_path

        def fit(self, X_train, y_train):
            return self

        def predict(self, X_test):
            return pd.Series([321.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    model, predictions, metadata = run_tabpfn_regression(X_train, y_train, X_test, device="gpu")

    assert not hasattr(model, "device")
    assert list(predictions) == [321.0]
    assert "did not accept" in metadata["notes"].lower()


def test_run_tabpfn_regression_rejects_invalid_direct_model_path_before_instantiation(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    instantiated = {"value": False}

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            instantiated["value"] = True

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(ValueError, match=r"\.ckpt"):
        run_tabpfn_regression(X_train, y_train, X_test, model_path="mystery")

    assert instantiated["value"] is False


def test_run_tabpfn_regression_preserves_safe_label_for_windows_paths(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            self.random_state = random_state
            self.model_path = model_path

        def fit(self, X_train, y_train):
            return self

        def predict(self, X_test):
            return pd.Series([123.0], index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    X_train = pd.DataFrame({"feature": [1.0]})
    y_train = pd.Series([10.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    _, _, metadata = run_tabpfn_regression(X_train, y_train, X_test, model_path=r"C:\temp\tabpfn\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt")

    assert metadata["model_path"] == "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"
    assert r"C:\temp\tabpfn\tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt" not in metadata["notes"]


def test_run_autogluon_regression_uses_default_and_custom_artifact_dirs(monkeypatch):
    from elferspot_listings.modeling.challengers import run_autogluon_regression

    class FakeTabularPredictor:
        created = []

        def __init__(self, label, path, problem_type):
            self.label = label
            self.path = path
            self.problem_type = problem_type
            self.fit_args = None
            self.predict_args = None
            self.leaderboard_args = None
            FakeTabularPredictor.created.append(self)

        def fit(self, train_df, time_limit, presets):
            self.fit_args = (train_df.copy(), time_limit, presets)
            return self

        def predict(self, features):
            self.predict_args = features.copy()
            return pd.Series([654.0], index=features.index)

        def leaderboard(self, data, silent):
            self.leaderboard_args = (data.copy(), silent)
            return pd.DataFrame({"model": ["fake"], "score": [0.0]})

    fake_autogluon = types.ModuleType("autogluon")
    fake_autogluon.__path__ = []
    fake_tabular = types.ModuleType("autogluon.tabular")
    fake_tabular.TabularPredictor = FakeTabularPredictor
    fake_autogluon.tabular = fake_tabular
    monkeypatch.setitem(sys.modules, "autogluon", fake_autogluon)
    monkeypatch.setitem(sys.modules, "autogluon.tabular", fake_tabular)

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with tempfile.TemporaryDirectory() as tmpdir:
        temp_path = Path(tmpdir)
        output_dir = temp_path / "results"
        custom_artifact_dir = temp_path / "custom-autogluon"

        default_predictor, default_predictions, default_leaderboard, default_metadata = run_autogluon_regression(
            train_df,
            test_df,
            "price_in_eur",
            output_dir,
            time_limit=15,
        )
        custom_predictor, custom_predictions, custom_leaderboard, custom_metadata = run_autogluon_regression(
            train_df,
            test_df,
            "price_in_eur",
            output_dir,
            time_limit=15,
            artifact_dir=custom_artifact_dir,
        )

        assert default_predictor.path == str(output_dir / "autogluon")
        assert default_metadata["model_name"] == "autogluon"
        assert default_metadata["time_limit_seconds"] == 15
        assert default_metadata["presets"] == "best_quality"
        assert default_metadata["runtime_seconds"] >= 0
        assert list(default_predictions) == [654.0]
        assert list(default_leaderboard["model"]) == ["fake"]
        assert (output_dir / "autogluon" / "leaderboard.csv").exists()

        assert custom_predictor.path == str(custom_artifact_dir)
        assert custom_metadata["model_name"] == "autogluon"
        assert custom_metadata["time_limit_seconds"] == 15
        assert custom_metadata["presets"] == "best_quality"
        assert custom_metadata["runtime_seconds"] >= 0
        assert list(custom_predictions) == [654.0]
        assert list(custom_leaderboard["model"]) == ["fake"]
        assert (custom_artifact_dir / "leaderboard.csv").exists()
        assert len(FakeTabularPredictor.created) == 2
