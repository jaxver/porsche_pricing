from __future__ import annotations

import json
import importlib.util
import sys
import types
from pathlib import Path
from sklearn.dummy import DummyRegressor

import config
import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from elferspot_listings.modeling import benchmark_db
from elferspot_listings.modeling.baselines import MedianRegressor
from elferspot_listings.modeling.challengers import OptionalDependencyNotInstalledError
from elferspot_listings.modeling.persistence import SkopsNotInstalledError
from elferspot_listings.modeling.train import train_baseline_models


def _gold_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "price_in_eur": [100000.0, 120000.0, 140000.0, 160000.0, 180000.0, 200000.0, 220000.0, 240000.0],
            "Mileage_km": [10000.0, 20000.0, 30000.0, 40000.0, 50000.0, 60000.0, 70000.0, 80000.0],
            "Year of construction": [1995, 1997, 2000, 2003, 2005, 2008, 2011, 2014],
            "model_category": ["911", "911", "Cayenne", "Boxster", "Targa", "SUV", "964", "992"],
        }
    )


def _expected_holdout_indices(frame: pd.DataFrame) -> list[int]:
    train_index, test_index = train_test_split(frame.index, test_size=0.25, random_state=42)
    return list(test_index)


def test_train_baseline_models_with_ridge_only_runs_ridge(tmp_path, monkeypatch):
    monkeypatch.setattr("elferspot_listings.modeling.train.MedianRegressor", lambda: (_ for _ in ()).throw(AssertionError("median should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_elasticnet_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("elasticnet should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skrub_ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("xgboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_ridge_pipeline", lambda _selected: DummyRegressor(strategy="mean"))

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=42, models=["ridge"])

    assert set(result.metrics) == {"ridge"}
    assert set(result.predictions["model_name"]) == {"ridge"}
    assert result.predictions["row_index"].tolist() == _expected_holdout_indices(_gold_frame())


def test_train_baseline_models_rejects_invalid_autogluon_dynamic_stacking_even_without_autogluon(tmp_path, monkeypatch):
    monkeypatch.setattr("elferspot_listings.modeling.train.MedianRegressor", lambda: (_ for _ in ()).throw(AssertionError("median should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_elasticnet_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("elasticnet should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skrub_ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("xgboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_ridge_pipeline", lambda _selected: DummyRegressor(strategy="mean"))

    with pytest.raises((TypeError, ValueError), match="autogluon_dynamic_stacking"):
        train_baseline_models(
            _gold_frame(),
            tmp_path,
            random_state=42,
            models=["ridge"],
            autogluon_dynamic_stacking="auto",  # type: ignore[arg-type]
        )


def test_train_baseline_models_with_xgboost_only_runs_xgboost_without_boolean_flag(tmp_path, monkeypatch):
    monkeypatch.setattr("elferspot_listings.modeling.train.MedianRegressor", lambda: (_ for _ in ()).throw(AssertionError("median should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_elasticnet_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("elasticnet should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skrub_ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda _selected, random_state=42: DummyRegressor(strategy="mean"))

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=42, models=["xgboost"])

    assert set(result.metrics) == {"xgboost"}
    assert set(result.predictions["model_name"]) == {"xgboost"}


def test_train_baseline_models_with_perpetual_only_runs_perpetual(tmp_path, monkeypatch):
    monkeypatch.setattr("elferspot_listings.modeling.train.MedianRegressor", lambda: (_ for _ in ()).throw(AssertionError("median should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_elasticnet_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("elasticnet should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skrub_ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("xgboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_perpetual_pipeline", lambda _selected, random_state=42: DummyRegressor(strategy="mean"))

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=42, models=["perpetual"])

    assert set(result.metrics) == {"perpetual"}
    assert set(result.predictions["model_name"]) == {"perpetual"}


def test_train_baseline_models_with_tabfm_only_runs_tabfm(tmp_path, monkeypatch):
    monkeypatch.setattr("elferspot_listings.modeling.train.MedianRegressor", lambda: (_ for _ in ()).throw(AssertionError("median should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_elasticnet_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("elasticnet should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("skrub_ridge should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("xgboost should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.fit_catboost_regressor", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("catboost should not run")))

    def fake_tabfm(X_train, y_train, X_test, random_state=42):
        return object(), pd.Series([777.0] * len(X_test), index=X_test.index), {
            "model_name": "tabfm",
            "backend": "pytorch",
            "runtime_seconds": 0.0,
            "notes": "fake tabfm note",
        }

    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabfm_regression", fake_tabfm)

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=42, models=["tabfm"])

    assert set(result.metrics) == {"tabfm"}
    assert set(result.predictions["model_name"]) == {"tabfm"}
    assert result.predictions["predicted_price_eur"].tolist() == [777.0] * len(result.predictions)


def test_train_baseline_models_records_missing_perpetual_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.build_perpetual_pipeline", lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("perpetual is not installed")))

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_perpetual=True)

    assert "perpetual" not in result.metrics
    assert result.skipped_models.get("perpetual") == "perpetual is not installed"


def test_train_baseline_models_with_only_missing_perpetual_writes_empty_outputs(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr(
        "elferspot_listings.modeling.train.build_perpetual_pipeline",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(ImportError("perpetual is not installed")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, models=["perpetual"])

    assert result.metrics == {}
    assert result.skipped_models == {"perpetual": "perpetual is not installed"}
    assert list(result.predictions.columns) == [
        "row_index",
        "model_name",
        "actual_price_eur",
        "predicted_price_eur",
        "residual_eur",
    ]
    assert result.predictions.empty
    assert (tmp_path / "predictions.csv").exists()
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "MODEL_CARD.md").exists()
    assert (tmp_path / "skipped_models.json").exists()

    predictions_csv = (tmp_path / "predictions.csv").read_text(encoding="utf-8").splitlines()
    assert predictions_csv == ["row_index,model_name,actual_price_eur,predicted_price_eur,residual_eur"]


def test_train_baseline_models_rejects_invalid_model_name(tmp_path):
    with pytest.raises(ValueError, match="Unsupported model names"):
        train_baseline_models(_gold_frame(), tmp_path, models=["bogus"])


def test_train_baseline_models_writes_reports_and_returns_metrics(tmp_path, monkeypatch):
    skops_missing = importlib.util.find_spec("skops") is None

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)

    gold_df = _gold_frame()
    result = train_baseline_models(gold_df, tmp_path, random_state=42)
    expected_holdout = _expected_holdout_indices(gold_df)

    metrics_path = tmp_path / "metrics.json"
    predictions_path = tmp_path / "predictions.csv"

    assert metrics_path.exists()
    assert predictions_path.exists()
    assert result.skipped_models.get("skrub_ridge") == "skrub is not installed"
    if skops_missing:
        assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
    assert set(result.metrics) == {"median", "ridge", "elasticnet"}
    assert list(result.predictions.columns) == [
        "row_index",
        "model_name",
        "actual_price_eur",
        "predicted_price_eur",
        "residual_eur",
    ]
    assert set(result.predictions["model_name"]) == {"median", "ridge", "elasticnet"}
    assert result.predictions["model_name"].value_counts().to_dict() == {"median": 2, "ridge": 2, "elasticnet": 2}
    for model_name in ("median", "ridge", "elasticnet"):
        model_rows = result.predictions[result.predictions["model_name"] == model_name]
        assert model_rows["row_index"].tolist() == expected_holdout
        assert len(model_rows) == len(expected_holdout)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"median", "ridge", "elasticnet"}
    skipped_payload = json.loads((tmp_path / "skipped_models.json").read_text(encoding="utf-8"))
    assert skipped_payload.get("skrub_ridge") == "skrub is not installed"
    if skops_missing:
        assert skipped_payload.get("ridge_artifact") == "skops is not installed"


def test_train_baseline_models_can_tune_elasticnet_with_optuna(tmp_path, monkeypatch):
    captured: dict[str, object] = {}

    def fake_tune_elasticnet_params(X_train, y_train, selected, random_state=42, n_trials=25):
        captured["n_trials"] = n_trials
        captured["random_state"] = random_state
        captured["train_rows"] = len(X_train)
        return {"alpha": 0.01, "l1_ratio": 0.1, "max_iter": 5000}

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.tune_elasticnet_params", fake_tune_elasticnet_params)

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=17, tune_elasticnet=True, tuning_trials=7)

    assert captured == {"n_trials": 7, "random_state": 17, "train_rows": 6}
    assert "elasticnet" in result.metrics
    assert "elasticnet" in set(result.predictions["model_name"])


def test_train_baseline_models_logs_run_to_sqlite(tmp_path, monkeypatch):
    db_path = tmp_path / "benchmark_runs.db"
    known_sha = "0123456789abcdef0123456789abcdef01234567"

    monkeypatch.setattr(config, "BENCHMARK_DB", db_path)
    monkeypatch.setattr(benchmark_db, "_current_git_commit", lambda: known_sha)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.build_skrub_ridge_pipeline",
        lambda _selected: (_ for _ in ()).throw(ImportError("skrub is not installed")),
    )

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=17)

    latest = benchmark_db.get_latest_run(db_path)
    assert latest is not None
    assert latest["random_state"] == 17
    assert set(latest["metrics"]) == set(result.metrics)
    assert set(next(iter(latest["metrics"].values()))) == {"mae_eur", "median_ae", "mape", "within_10", "within_15"}
    assert latest["output_dir"] == str(tmp_path)
    assert latest["git_commit"] == known_sha
    assert latest["skipped"] == {"skrub_ridge": "skrub is not installed"}
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()


def test_train_baseline_models_logs_explicit_autogluon_run_flag(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    captured: dict[str, object] = {}

    def fake_insert_run(
        db_path,
        *,
        random_state,
        train_catboost,
        run_tabpfn,
        run_tabfm,
        run_autogluon,
        autogluon_tl,
        output_dir,
        duration_sec,
        git_commit=None,
    ):
        captured["run_autogluon"] = run_autogluon
        captured["run_tabpfn"] = run_tabpfn
        captured["run_tabfm"] = run_tabfm
        captured["random_state"] = random_state
        return 1

    monkeypatch.setattr(config, "BENCHMARK_DB", tmp_path / "benchmark_runs.db")
    monkeypatch.setattr(benchmark_db, "insert_run", fake_insert_run)
    monkeypatch.setattr(benchmark_db, "insert_metrics", lambda *_args, **_kwargs: None)
    monkeypatch.setattr(benchmark_db, "insert_skipped", lambda *_args, **_kwargs: None)
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda train_df, test_df, target, output_dir, **_kwargs: (
            object(),
            pd.Series([222.0] * len(test_df), index=test_df.index),
            pd.DataFrame({"model": ["fake"], "score": [0.5]}),
            {"model_name": "autogluon", "runtime_seconds": 0.0, "time_limit_seconds": 600, "presets": "best_quality", "dynamic_stacking": None},
        ),
    )

    train_baseline_models(gold_df, tmp_path, random_state=42, models=["autogluon"])

    assert captured["run_autogluon"] is True
    assert captured["run_tabpfn"] is False
    assert captured["run_tabfm"] is False


def test_train_baseline_models_ignores_benchmark_db_failures(tmp_path, monkeypatch):
    db_path = tmp_path / "benchmark_runs.db"

    monkeypatch.setattr(config, "BENCHMARK_DB", db_path)
    monkeypatch.setattr(benchmark_db, "insert_run", lambda *_args, **_kwargs: (_ for _ in ()).throw(RuntimeError("db down")))
    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())

    result = train_baseline_models(_gold_frame(), tmp_path, random_state=17)

    assert set(result.metrics) == {"median", "ridge", "elasticnet", "skrub_ridge"}
    assert (tmp_path / "metrics.json").exists()
    assert (tmp_path / "predictions.csv").exists()
    assert benchmark_db.get_latest_run(db_path) is None


def test_train_baseline_models_defaults_do_not_run_challengers(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_tabpfn_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run by default")),
    )
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_tabfm_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabfm should not run by default")),
    )
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run by default")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert set(result.metrics) == {"median", "ridge", "elasticnet", "skrub_ridge"}
    assert "tabpfn_default" not in result.metrics
    assert "tabfm" not in result.metrics
    assert "autogluon" not in result.metrics
    assert "tabpfn_default" not in set(result.predictions["model_name"])
    assert "tabfm" not in set(result.predictions["model_name"])
    assert "autogluon" not in set(result.predictions["model_name"])


def test_train_baseline_models_records_missing_tabpfn_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    class FakeTabPFNRegressor:
        def __init__(self, random_state=None, model_path=None):
            self.random_state = random_state
            self.model_path = model_path

        def fit(self, X_train, y_train):
            raise OSError("[WinError 10038] An operation was attempted on something that is not a socket")

        def predict(self, X_test):
            return pd.Series([111.0] * len(X_test), index=X_test.index)

    fake_tabpfn = types.ModuleType("tabpfn")
    fake_tabpfn.TabPFNRegressor = FakeTabPFNRegressor
    monkeypatch.setitem(sys.modules, "tabpfn", fake_tabpfn)

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run in this test")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabpfn=True)

    assert "tabpfn_default" not in result.metrics
    assert result.skipped_models.get("tabpfn_default") == (
        "TabPFN browser/license authentication failed. Accept the Prior Labs license in a browser manually, "
        "set `TABPFN_TOKEN` in the environment before rerunning, and avoid browser auth from proxied or "
        "non-interactive Windows runs."
    )


def test_train_baseline_models_records_missing_autogluon_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def fake_autogluon(*_args, **_kwargs):
        raise OptionalDependencyNotInstalledError("AutoGluon")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", fake_autogluon)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_tabpfn_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run in this test")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_autogluon=True)

    assert "autogluon" not in result.metrics
    assert result.skipped_models.get("autogluon") == 'Install AutoGluon with `python -m pip install -e ".[advanced]"`.'


def test_train_baseline_models_records_missing_tabpfn_client_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def fake_tabpfn_client(*_args, **_kwargs):
        raise OptionalDependencyNotInstalledError(
            "tabpfn-client",
            "tabpfn-client API authentication/access failed. Set or access your Prior Labs access token and retry.",
        )

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_client_regression", fake_tabpfn_client)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run in this test")),
    )

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=42,
        run_tabpfn=True,
        tabpfn_backend="client",
        tabpfn_thinking=True,
    )

    assert "tabpfn_client_thinking" not in result.metrics
    assert result.skipped_models.get("tabpfn_client_thinking") == (
        "tabpfn-client API authentication/access failed. Set or access your Prior Labs access token and retry."
    )


def test_train_baseline_models_records_tabpfn_cuda_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def fake_tabpfn(*_args, **_kwargs):
        raise OptionalDependencyNotInstalledError(
            "TabPFN",
            "local TabPFN GPU requested but CUDA is unavailable or Torch is not compiled with CUDA; rerun with `--device cpu` or install a CUDA-enabled PyTorch build.",
        )

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", fake_tabpfn)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run in this test")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabpfn=True, device="gpu")

    assert "tabpfn_default" not in result.metrics
    assert result.skipped_models.get("tabpfn_default") == (
        "local TabPFN GPU requested but CUDA is unavailable or Torch is not compiled with CUDA; rerun with `--device cpu` or install a CUDA-enabled PyTorch build."
    )


def test_train_baseline_models_records_missing_tabfm_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def fake_tabfm(*_args, **_kwargs):
        raise OptionalDependencyNotInstalledError("TabFM", "Install TabFM with `python -m pip install -e \".[advanced]\"`.")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabfm_regression", fake_tabfm)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run in this test")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabfm=True)

    assert "tabfm" not in result.metrics
    assert result.skipped_models.get("tabfm") == 'Install TabFM with `python -m pip install -e "[advanced]"`.'


def test_train_baseline_models_appends_tabfm_predictions_when_enabled(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_tabfm(X_train, y_train, X_test, random_state=42):
        captured["X_train"] = X_train.copy()
        captured["X_test"] = X_test.copy()
        captured["random_state"] = random_state
        return object(), pd.Series([444.0] * len(X_test), index=X_test.index), {
            "model_name": "tabfm",
            "backend": "pytorch",
            "runtime_seconds": 0.0,
            "notes": "fake tabfm note",
        }

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabfm_regression", fake_tabfm)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabfm=True)

    assert "tabfm" in result.metrics
    assert "tabfm" in set(result.predictions["model_name"])
    assert result.predictions[result.predictions["model_name"] == "tabfm"]["predicted_price_eur"].tolist() == [444.0] * len(
        result.predictions[result.predictions["model_name"] == "tabfm"]
    )
    assert captured["random_state"] == 42
    assert not captured["X_train"].isna().any().any()
    assert not captured["X_test"].isna().any().any()


def test_train_baseline_models_records_missing_xgboost_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def raise_import_error(_selected, random_state=42):
        raise ImportError("xgboost is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", raise_import_error)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_xgboost=True)

    assert "xgboost" not in result.metrics
    assert result.skipped_models.get("xgboost") == "xgboost is not installed"


def test_train_baseline_models_appends_xgboost_predictions_when_enabled(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.build_xgboost_pipeline", lambda _selected, random_state=42: MedianRegressor())

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_xgboost=True)

    assert "xgboost" in result.metrics
    assert "xgboost" in set(result.predictions["model_name"])


def test_train_baseline_models_appends_tabpfn_predictions_with_encoded_features(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_tabpfn(X_train, y_train, X_test, random_state=42, model_path=None, model_name="tabpfn_default"):
        captured["X_train"] = X_train.copy()
        captured["X_test"] = X_test.copy()
        captured["random_state"] = random_state
        captured["model_path"] = model_path
        captured["model_name"] = model_name
        return object(), pd.Series([111.0] * len(X_test), index=X_test.index), {"model_name": model_name, "model_path": model_path, "runtime_seconds": 0.0, "notes": "fake checkpoint note"}

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", fake_tabpfn)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabpfn=True)

    assert "tabpfn_default" in result.metrics
    assert "tabpfn_default" in set(result.predictions["model_name"])
    assert result.predictions[result.predictions["model_name"] == "tabpfn_default"]["predicted_price_eur"].tolist() == [111.0] * len(
        result.predictions[result.predictions["model_name"] == "tabpfn_default"]
    )
    assert captured["random_state"] == 42
    assert captured["model_path"] is None
    assert captured["model_name"] == "tabpfn_default"
    assert not captured["X_train"].isna().any().any()
    assert not captured["X_test"].isna().any().any()
    assert any(column.endswith("_nan") for column in captured["X_train"].columns)


def test_train_baseline_models_routes_tabpfn_client_backend_and_thinking_mode(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_tabpfn_client(X_train, y_train, X_test, random_state=42, thinking_mode=False, thinking_effort="medium", thinking_metric="rmse", thinking_timeout_s=None):
        captured["X_train"] = X_train.copy()
        captured["X_test"] = X_test.copy()
        captured["kwargs"] = {
            "random_state": random_state,
            "thinking_mode": thinking_mode,
            "thinking_effort": thinking_effort,
            "thinking_metric": thinking_metric,
            "thinking_timeout_s": thinking_timeout_s,
        }
        return object(), pd.Series([333.0] * len(X_test), index=X_test.index), {
            "model_name": "tabpfn_client_thinking",
            "backend": "client",
            "runtime_seconds": 0.0,
            "notes": "backend=client effort=high metric=mae timeout=12",
        }

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_client_regression", fake_tabpfn_client)
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local tabpfn should not run")))

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=11,
        run_tabpfn=True,
        tabpfn_backend="client",
        tabpfn_thinking=True,
        tabpfn_thinking_effort="high",
        tabpfn_thinking_metric="mae",
        tabpfn_thinking_timeout=12,
    )

    assert captured["kwargs"] == {
        "random_state": 11,
        "thinking_mode": True,
        "thinking_effort": "high",
        "thinking_metric": "mae",
        "thinking_timeout_s": 12,
    }
    assert "tabpfn_client_thinking" in result.metrics
    assert "tabpfn_client_thinking" in set(result.predictions["model_name"])
    assert not captured["X_train"].isna().any().any()
    assert not captured["X_test"].isna().any().any()


def test_train_baseline_models_ignores_tabpfn_client_checkpoint_when_tabpfn_is_not_selected(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("local tabpfn should not run")))
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_client_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("client tabpfn should not run")))

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=11,
        models=["ridge"],
        tabpfn_backend="client",
        tabpfn_model_paths=["default"],
    )

    assert set(result.metrics) == {"ridge"}
    assert "tabpfn" not in result.metrics
    assert "tabpfn_client" not in result.metrics


def test_train_baseline_models_appends_explicit_tabpfn_checkpoint_variants(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured: list[tuple[str, str | None]] = []

    def fake_tabpfn(X_train, y_train, X_test, random_state=42, model_path=None, model_name="tabpfn_default"):
        captured.append((model_name, model_path))
        return object(), pd.Series([111.0] * len(X_test), index=X_test.index), {
            "model_name": model_name,
            "model_path": model_path,
            "runtime_seconds": 0.0,
            "notes": "fake checkpoint note",
        }

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", fake_tabpfn)

    result = train_baseline_models(
        gold_df,
        tmp_path,
        random_state=42,
        run_tabpfn=True,
        tabpfn_model_paths=["default", "mediumdata", "ood"],
    )

    assert captured == [
        ("tabpfn_default", None),
        ("tabpfn_mediumdata", "tabpfn-v3-regressor-v3_20260417_mediumdata.ckpt"),
        ("tabpfn_ood", "tabpfn-v3-regressor-v3_20260506_ood.ckpt"),
    ]
    assert {"tabpfn_default", "tabpfn_mediumdata", "tabpfn_ood"}.issubset(result.metrics)
    assert {"tabpfn_default", "tabpfn_mediumdata", "tabpfn_ood"}.issubset(set(result.predictions["model_name"]))


def test_normalize_tabpfn_checkpoint_alias_rejects_unknown_aliases():
    from elferspot_listings.modeling.train import _normalize_tabpfn_checkpoint_alias

    with pytest.raises(ValueError, match="Supported aliases"):
        _normalize_tabpfn_checkpoint_alias("mystery")


def test_train_baseline_models_appends_autogluon_predictions_and_uses_target_frame(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_autogluon(
        train_df,
        test_df,
        target,
        output_dir,
        time_limit=600,
        artifact_dir=None,
        presets="best_quality",
        dynamic_stacking=None,
        clean_output=False,
    ):
        captured["train_df"] = train_df.copy()
        captured["test_df"] = test_df.copy()
        captured["target"] = target
        captured["output_dir"] = Path(output_dir)
        captured["time_limit"] = time_limit
        captured["artifact_dir"] = artifact_dir
        captured["presets"] = presets
        captured["dynamic_stacking"] = dynamic_stacking
        captured["clean_output"] = clean_output
        return (
            object(),
            pd.Series([222.0] * len(test_df), index=test_df.index),
            pd.DataFrame({"model": ["fake"], "score": [0.5]}),
            {
                "model_name": "autogluon",
                "runtime_seconds": 0.0,
                "time_limit_seconds": time_limit,
                "presets": presets,
                "dynamic_stacking": dynamic_stacking,
            },
        )

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", fake_autogluon)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_autogluon=True, autogluon_time_limit=33)

    assert "autogluon" in result.metrics
    assert "autogluon" in set(result.predictions["model_name"])
    assert result.predictions[result.predictions["model_name"] == "autogluon"]["predicted_price_eur"].tolist() == [222.0] * len(
        result.predictions[result.predictions["model_name"] == "autogluon"]
    )
    assert captured["target"] == "price_in_eur"
    assert captured["time_limit"] == 33
    assert captured["presets"] == "best_quality"
    assert captured["dynamic_stacking"] is None
    assert captured["clean_output"] is False
    assert set(captured["train_df"].columns) == set(gold_df.columns)
    assert set(captured["test_df"].columns) == set(gold_df.columns)
    assert "price_in_eur" in captured["train_df"].columns
    assert "price_in_eur" in captured["test_df"].columns


def test_train_baseline_models_forwards_explicit_autogluon_dynamic_stacking_bool(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_autogluon(
        train_df,
        test_df,
        target,
        output_dir,
        time_limit=600,
        artifact_dir=None,
        presets="best_quality",
        dynamic_stacking=None,
        clean_output=False,
    ):
        captured["dynamic_stacking"] = dynamic_stacking
        return (
            object(),
            pd.Series([222.0] * len(test_df), index=test_df.index),
            pd.DataFrame({"model": ["fake"], "score": [0.5]}),
            {
                "model_name": "autogluon",
                "runtime_seconds": 0.0,
                "time_limit_seconds": time_limit,
                "presets": presets,
                "dynamic_stacking": dynamic_stacking,
            },
        )

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_autogluon_regression", fake_autogluon)

    train_baseline_models(gold_df, tmp_path, random_state=42, run_autogluon=True, autogluon_dynamic_stacking=True)

    assert captured["dynamic_stacking"] is True


def test_prepare_tabpfn_features_ignores_test_only_categories():
    from elferspot_listings.modeling.train import _prepare_tabpfn_features

    X_train = pd.DataFrame(
        {
            "model_category": ["911", "Cayman"],
            "color": ["red", None],
            "Mileage_km": [10000.0, None],
        },
        index=[10, 11],
    )
    X_test = pd.DataFrame(
        {
            "model_category": ["911 GT3"],
            "color": ["red"],
            "Mileage_km": [30000.0],
        },
        index=[20],
    )

    prepared_train, prepared_test = _prepare_tabpfn_features(X_train, X_test)

    assert set(prepared_train.columns) == set(prepared_test.columns)
    assert not any("911 GT3" in column for column in prepared_train.columns)
    assert prepared_train.index.tolist() == [10, 11]
    assert prepared_test.index.tolist() == [20]
    assert not prepared_train.isna().any().any()
    assert not prepared_test.isna().any().any()


def test_train_baseline_models_cleans_stale_autogluon_output_when_not_running(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    stale_dir = tmp_path / "autogluon"
    stale_dir.mkdir(parents=True)
    (stale_dir / "leaderboard.csv").write_text("stale", encoding="utf-8")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_autogluon=False)

    assert "autogluon" not in result.metrics
    assert not stale_dir.exists()


def test_train_baseline_models_rejects_unsafe_autogluon_cleanup_path(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    autogluon_target = tmp_path / "autogluon"
    broken_target = tmp_path / "missing-target"

    try:
        autogluon_target.symlink_to(broken_target, target_is_directory=True)
    except OSError as exc:
        pytest.skip(f"symlink creation is unavailable in this environment: {exc}")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())

    with pytest.raises(ValueError, match="dedicated AutoGluon"):
        train_baseline_models(gold_df, tmp_path, random_state=42, run_autogluon=False)

    assert autogluon_target.is_symlink()
    assert not autogluon_target.exists()



def test_train_baseline_models_clears_stale_skipped_models_file_when_skrub_recovers(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    train_baseline_models(gold_df, tmp_path, random_state=42)

    skipped_path = tmp_path / "skipped_models.json"
    first_payload = json.loads(skipped_path.read_text(encoding="utf-8"))
    assert first_payload.get("skrub_ridge") == "skrub is not installed"

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert result.skipped_models.get("skrub_ridge") is None
    if skipped_path.exists():
        second_payload = json.loads(skipped_path.read_text(encoding="utf-8"))
        assert second_payload.get("skrub_ridge") is None
    assert set(result.metrics) == {"median", "ridge", "elasticnet", "skrub_ridge"}


def test_train_baseline_models_removes_stale_sklearn_artifacts_when_skops_is_unavailable(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ridge.skops").write_text("stale ridge", encoding="utf-8")
    (artifacts_dir / "elasticnet.skops").write_text("stale elasticnet", encoding="utf-8")
    (artifacts_dir / "xgboost.skops").write_text("stale xgboost", encoding="utf-8")
    (artifacts_dir / "skrub_ridge.skops").write_text("stale skrub", encoding="utf-8")

    def raise_skops_missing(*_args, **_kwargs):
        raise SkopsNotInstalledError("skops is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", raise_skops_missing)

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert not (artifacts_dir / "ridge.skops").exists()
    assert not (artifacts_dir / "elasticnet.skops").exists()
    assert not (artifacts_dir / "xgboost.skops").exists()
    assert not (artifacts_dir / "skrub_ridge.skops").exists()
    assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
    assert result.skipped_models.get("elasticnet_artifact") == "skops is not installed"
    assert result.skipped_models.get("skrub_ridge_artifact") == "skops is not installed"


def test_train_baseline_models_records_non_skops_artifact_failure_reason(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ridge.skops").write_text("stale ridge", encoding="utf-8")

    def raise_value_error(*_args, **_kwargs):
        raise ValueError("serializer broke")

    def raise_import_error(_selected):
        raise ImportError("skrub is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", raise_import_error)
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", raise_value_error)

    try:
        train_baseline_models(gold_df, tmp_path, random_state=42)
    except ValueError as exc:
        assert str(exc) == "serializer broke"
    else:
        raise AssertionError("train_baseline_models should fail on non-skops persistence errors")

    assert not (artifacts_dir / "ridge.skops").exists()


def test_train_baseline_models_rolls_back_previous_artifacts_on_later_save_failure(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    unrelated_path = artifacts_dir / "notes.txt"
    unrelated_path.write_text("keep me", encoding="utf-8")

    calls: list[str] = []

    def fake_save(model, path):
        calls.append(Path(path).name)
        Path(path).write_text("artifact", encoding="utf-8")
        if len(calls) == 2:
            raise ValueError("serializer broke late")
        return Path(path)

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", fake_save)

    try:
        train_baseline_models(gold_df, tmp_path, random_state=42)
    except ValueError as exc:
        assert str(exc) == "serializer broke late"
    else:
        raise AssertionError("train_baseline_models should fail on unexpected late persistence errors")

    assert calls == ["ridge.skops", "elasticnet.skops"]
    assert not (artifacts_dir / "ridge.skops").exists()
    assert not (artifacts_dir / "elasticnet.skops").exists()
    assert not (artifacts_dir / "skrub_ridge.skops").exists()
    assert unrelated_path.exists()
