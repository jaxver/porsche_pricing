from __future__ import annotations

import json
import importlib.util
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

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
    assert set(result.metrics) == {"median", "ridge"}
    assert list(result.predictions.columns) == [
        "row_index",
        "model_name",
        "actual_price_eur",
        "predicted_price_eur",
        "residual_eur",
    ]
    assert set(result.predictions["model_name"]) == {"median", "ridge"}
    assert result.predictions["model_name"].value_counts().to_dict() == {"median": 2, "ridge": 2}
    for model_name in ("median", "ridge"):
        model_rows = result.predictions[result.predictions["model_name"] == model_name]
        assert model_rows["row_index"].tolist() == expected_holdout
        assert len(model_rows) == len(expected_holdout)

    metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
    assert set(metrics) == {"median", "ridge"}
    skipped_payload = json.loads((tmp_path / "skipped_models.json").read_text(encoding="utf-8"))
    assert skipped_payload.get("skrub_ridge") == "skrub is not installed"
    if skops_missing:
        assert skipped_payload.get("ridge_artifact") == "skops is not installed"


def test_train_baseline_models_defaults_do_not_run_challengers(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_tabpfn_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabpfn should not run by default")),
    )
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run by default")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert set(result.metrics) == {"median", "ridge", "skrub_ridge"}
    assert "tabpfn" not in result.metrics
    assert "autogluon" not in result.metrics
    assert "tabpfn" not in set(result.predictions["model_name"])
    assert "autogluon" not in set(result.predictions["model_name"])


def test_train_baseline_models_records_missing_tabpfn_skip_and_continues(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    def fake_tabpfn(*_args, **_kwargs):
        raise OptionalDependencyNotInstalledError("TabPFN")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", fake_tabpfn)
    monkeypatch.setattr(
        "elferspot_listings.modeling.train.run_autogluon_regression",
        lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("autogluon should not run in this test")),
    )

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabpfn=True)

    assert "tabpfn" not in result.metrics
    assert result.skipped_models.get("tabpfn") == "Install TabPFN with `python -m pip install -r requirements-advanced.txt`."


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
    assert result.skipped_models.get("autogluon") == "Install AutoGluon with `python -m pip install -r requirements-advanced.txt`."


def test_train_baseline_models_appends_tabpfn_predictions_with_encoded_features(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_tabpfn(X_train, y_train, X_test, random_state=42):
        captured["X_train"] = X_train.copy()
        captured["X_test"] = X_test.copy()
        captured["random_state"] = random_state
        return object(), pd.Series([111.0] * len(X_test), index=X_test.index), {"model_name": "tabpfn", "runtime_seconds": 0.0, "notes": "fake checkpoint note"}

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabpfn_regression", fake_tabpfn)

    result = train_baseline_models(gold_df, tmp_path, random_state=42, run_tabpfn=True)

    assert "tabpfn" in result.metrics
    assert "tabpfn" in set(result.predictions["model_name"])
    assert result.predictions[result.predictions["model_name"] == "tabpfn"]["predicted_price_eur"].tolist() == [111.0] * len(
        result.predictions[result.predictions["model_name"] == "tabpfn"]
    )
    assert captured["random_state"] == 42
    assert not captured["X_train"].isna().any().any()
    assert not captured["X_test"].isna().any().any()
    assert any(column.endswith("_nan") for column in captured["X_train"].columns)


def test_train_baseline_models_appends_autogluon_predictions_and_uses_target_frame(tmp_path, monkeypatch):
    gold_df = _gold_frame()

    captured = {}

    def fake_autogluon(train_df, test_df, target, output_dir, time_limit=600, artifact_dir=None):
        captured["train_df"] = train_df.copy()
        captured["test_df"] = test_df.copy()
        captured["target"] = target
        captured["output_dir"] = Path(output_dir)
        captured["time_limit"] = time_limit
        captured["artifact_dir"] = artifact_dir
        return object(), pd.Series([222.0] * len(test_df), index=test_df.index), pd.DataFrame({"model": ["fake"], "score": [0.5]}), {"model_name": "autogluon", "runtime_seconds": 0.0, "time_limit_seconds": time_limit, "presets": "best_quality"}

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
    assert set(captured["train_df"].columns) == set(gold_df.columns)
    assert set(captured["test_df"].columns) == set(gold_df.columns)
    assert "price_in_eur" in captured["train_df"].columns
    assert "price_in_eur" in captured["test_df"].columns


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
    assert set(result.metrics) == {"median", "ridge", "skrub_ridge"}


def test_train_baseline_models_removes_stale_sklearn_artifacts_when_skops_is_unavailable(tmp_path, monkeypatch):
    gold_df = _gold_frame()
    artifacts_dir = tmp_path / "artifacts"
    artifacts_dir.mkdir(parents=True)
    (artifacts_dir / "ridge.skops").write_text("stale ridge", encoding="utf-8")
    (artifacts_dir / "skrub_ridge.skops").write_text("stale skrub", encoding="utf-8")

    def raise_skops_missing(*_args, **_kwargs):
        raise SkopsNotInstalledError("skops is not installed")

    monkeypatch.setattr("elferspot_listings.modeling.train.build_skrub_ridge_pipeline", lambda _selected: MedianRegressor())
    monkeypatch.setattr("elferspot_listings.modeling.train.save_sklearn_model", raise_skops_missing)

    result = train_baseline_models(gold_df, tmp_path, random_state=42)

    assert not (artifacts_dir / "ridge.skops").exists()
    assert not (artifacts_dir / "skrub_ridge.skops").exists()
    assert result.skipped_models.get("ridge_artifact") == "skops is not installed"
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

    assert calls == ["ridge.skops", "skrub_ridge.skops"]
    assert not (artifacts_dir / "ridge.skops").exists()
    assert not (artifacts_dir / "skrub_ridge.skops").exists()
    assert unrelated_path.exists()
