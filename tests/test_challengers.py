from __future__ import annotations

import importlib
import importlib.util
import sys
import types
import tempfile
from pathlib import Path

import pandas as pd
import pytest


def test_challengers_module_imports_without_advanced_dependencies():
    module = importlib.import_module("elferspot_listings.modeling.challengers")

    assert hasattr(module, "run_tabpfn_regression")
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

    with pytest.raises(RuntimeError, match=r"requirements-advanced\.txt"):
        run_tabpfn_regression(X_train, y_train, X_test)


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

    with pytest.raises(RuntimeError, match=r"requirements-advanced\.txt"):
        run_autogluon_regression(train_df, test_df, "price_in_eur", ".")


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
