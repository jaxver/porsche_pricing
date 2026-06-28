from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pandas as pd


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
        "tabpfn_model_paths": ["mediumdata", "ood"],
        "run_autogluon": True,
        "autogluon_time_limit": 33,
    }
    assert printed == {
        "metrics": {"ridge": {"mae_eur": 123.4}},
        "output_dir": str(tmp_path / "benchmarks" / "cli_run"),
        "skipped_models": {"xgboost": "missing"},
    }
