from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd
import pytest


def test_cli_parses_arguments_and_prints_json(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df if Path(path) == tmp_path / "default_input.xlsx" else (_ for _ in ()).throw(AssertionError("unexpected input path")))

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["gold_df"] = gold_df_arg.copy()
        captured["output_dir"] = Path(output_dir)
        captured["kwargs"] = kwargs
        return SimpleNamespace(
            metrics={"ridge": {"mae_eur": 123.4}},
            output_dir=Path(output_dir),
            skipped_models={"xgboost": "missing"},
        )

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(
        [
            "--model",
            "all",
            "--include-optionals",
            "--tune",
            "--random-state",
            "7",
            "--tuning-trials",
            "13",
            "--tabpfn-checkpoint",
            "mediumdata",
            "--tabpfn-checkpoint",
            "ood",
            "--autogluon-time-limit",
            "33",
            "--tabfm-n-estimators",
            "8",
            "--tabfm-batch-size",
            "2",
            "--tabfm-max-num-rows",
            "2000",
            "--tabfm-cv-folds",
            "5",
        ]
    )

    printed = json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["gold_df"].equals(gold_df)
    assert captured["output_dir"] == tmp_path / "benchmarks" / "cli_run"
    assert captured["kwargs"] == {
        "random_state": 7,
        "models": ["all"],
        "train_catboost": True,
        "tune_elasticnet": True,
        "tune_catboost": True,
        "tuning_trials": 13,
        "run_xgboost": True,
        "run_tabpfn": True,
        "run_tabfm": True,
        "run_perpetual": True,
        "tabfm_n_estimators": 8,
        "tabfm_batch_size": 2,
        "tabfm_max_num_rows": 2000,
        "tabfm_cv_folds": 5,
        "tabpfn_model_paths": ["mediumdata", "ood"],
        "tabpfn_backend": "local",
        "tabpfn_thinking": False,
        "tabpfn_thinking_effort": "medium",
        "tabpfn_thinking_timeout": None,
        "tabpfn_thinking_metric": "rmse",
        "run_autogluon": True,
        "autogluon_time_limit": 33,
        "autogluon_presets": "best_quality",
        "autogluon_dynamic_stacking": None,
        "autogluon_clean_output": False,
        "verbose": False,
    }
    assert printed == {
        "metrics": {"ridge": {"mae_eur": 123.4}},
        "output_dir": str(tmp_path / "benchmarks" / "cli_run"),
        "skipped_models": {"xgboost": "missing"},
    }


def test_cli_all_optionals_enables_tabpfn_without_checkpoint(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "all", "--include-optionals"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["run_tabpfn"] is True
    assert captured["kwargs"]["run_tabfm"] is True
    assert captured["kwargs"]["run_perpetual"] is True
    assert captured["kwargs"]["tabpfn_model_paths"] is None


def test_cli_perpetual_model_only_passes_only_that_model(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "perpetual"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"] == {
        "random_state": 42,
        "models": ["perpetual"],
        "train_catboost": False,
        "tune_elasticnet": False,
        "tune_catboost": False,
        "tuning_trials": 25,
        "run_xgboost": False,
        "run_tabpfn": False,
        "run_tabfm": False,
        "run_perpetual": False,
        "tabfm_n_estimators": None,
        "tabfm_batch_size": None,
        "tabfm_max_num_rows": None,
        "tabfm_cv_folds": None,
        "tabpfn_model_paths": None,
        "tabpfn_backend": "local",
        "tabpfn_thinking": False,
        "tabpfn_thinking_effort": "medium",
        "tabpfn_thinking_timeout": None,
        "tabpfn_thinking_metric": "rmse",
        "run_autogluon": False,
        "autogluon_time_limit": 600,
        "autogluon_presets": "best_quality",
        "autogluon_dynamic_stacking": None,
        "autogluon_clean_output": False,
        "verbose": False,
    }


def test_cli_passes_gpu_flags_to_train_baseline_models(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["gold_df"] = gold_df_arg.copy()
        captured["output_dir"] = Path(output_dir)
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "catboost", "--device", "gpu", "--gpu-devices", "0"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["device"] == "gpu"
    assert captured["kwargs"]["gpu_devices"] == "0"


def test_cli_passes_verbose_flag_to_train_baseline_models(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "ridge", "--verbose"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["verbose"] is True


def test_cli_passes_tabpfn_client_thinking_kwargs(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(
        [
            "--model",
            "tabpfn",
            "--tabpfn-backend",
            "client",
            "--tabpfn-thinking",
            "--tabpfn-thinking-effort",
            "high",
            "--tabpfn-thinking-timeout",
            "12",
            "--tabpfn-thinking-metric",
            "mae",
        ]
    )
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"] == {
        "random_state": 42,
        "models": ["tabpfn"],
        "train_catboost": False,
        "tune_elasticnet": False,
        "tune_catboost": False,
        "tuning_trials": 25,
        "run_xgboost": False,
        "run_tabpfn": False,
        "run_tabfm": False,
        "run_perpetual": False,
        "tabfm_n_estimators": None,
        "tabfm_batch_size": None,
        "tabfm_max_num_rows": None,
        "tabfm_cv_folds": None,
        "tabpfn_model_paths": None,
        "tabpfn_backend": "client",
        "tabpfn_thinking": True,
        "tabpfn_thinking_effort": "high",
        "tabpfn_thinking_timeout": 12,
        "tabpfn_thinking_metric": "mae",
        "run_autogluon": False,
        "autogluon_time_limit": 600,
        "autogluon_presets": "best_quality",
        "autogluon_dynamic_stacking": None,
        "autogluon_clean_output": False,
        "verbose": False,
    }


@pytest.mark.parametrize(
    ("cli_value", "expected"),
    [("true", True), ("false", False)],
)
def test_cli_converts_autogluon_dynamic_stacking_strings(monkeypatch, capsys, tmp_path, cli_value, expected):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "autogluon", "--autogluon-dynamic-stacking", cli_value])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["autogluon_dynamic_stacking"] is expected


def test_cli_rejects_tabpfn_thinking_with_local_backend(monkeypatch, tmp_path):
    from elferspot_listings.modeling import cli

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)

    try:
        cli.main(["--model", "tabpfn", "--tabpfn-backend", "local", "--tabpfn-thinking"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("cli.main should reject thinking mode with local backend")


def test_cli_rejects_tabpfn_checkpoint_with_client_backend(monkeypatch, tmp_path):
    from elferspot_listings.modeling import cli

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)

    try:
        cli.main(["--model", "tabpfn", "--tabpfn-backend", "client", "--tabpfn-checkpoint", "default"])
    except SystemExit as exc:
        assert exc.code == 2
    else:
        raise AssertionError("cli.main should reject checkpoints with client backend")


def test_cli_accepts_tabfm_model_selection(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "tabfm"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["models"] == ["tabfm"]


def test_cli_accepts_high_price_specialist_model_selection(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "high_price_specialist"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["models"] == ["high_price_specialist"]


def test_cli_accepts_stacked_ensemble_model_selection(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "stacked_ensemble"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["models"] == ["stacked_ensemble"]


def test_cli_allows_ridge_with_client_backend_checkpoint_without_selecting_tabpfn(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    gold_df = pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]})

    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: gold_df)

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "ridge", "--tabpfn-backend", "client", "--tabpfn-checkpoint", "default"])
    json.loads(capsys.readouterr().out)

    assert exit_code == 0
    assert captured["kwargs"]["models"] == ["ridge"]
    assert captured["kwargs"]["run_tabpfn"] is False
    assert captured["kwargs"]["tabpfn_model_paths"] == ["default"]
