from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import pandas as pd

from .train import train_baseline_models


MODEL_CHOICES = ["median", "ridge", "elasticnet", "skrub_ridge", "xgboost", "catboost", "tabpfn", "autogluon", "all"]


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run model benchmarks and print JSON metrics.")
    parser.add_argument("--model", action="append", choices=MODEL_CHOICES, default=None, help="Model to run; repeat for multiple selections.")
    parser.add_argument("--input", type=Path, default=Path(config.LISTINGS_GOLD), help="Gold-layer listings input file.")
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path(config.RESULTS_DIR) / "benchmarks" / "cli_run",
        help="Directory for benchmark outputs.",
    )
    parser.add_argument("--random-state", type=int, default=42, help="Random seed for the train/test split.")
    parser.add_argument("--tune", action="store_true", help="Tune ElasticNet and CatBoost when those models are selected.")
    parser.add_argument("--tuning-trials", type=int, default=25, help="Number of Optuna trials for tuning.")
    parser.add_argument(
        "--tabpfn-checkpoint",
        action="append",
        default=None,
        help="TabPFN checkpoint alias or .ckpt path; repeat to run multiple checkpoints.",
    )
    parser.add_argument("--autogluon-time-limit", type=int, default=600, help="Time limit in seconds for AutoGluon.")
    parser.add_argument("--include-optionals", action="store_true", help="Run optional models when used together with --model all.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)

    gold_df = pd.read_excel(args.input)
    models = args.model or ["all"]
    model_set = set(models)
    include_optionals = args.include_optionals and "all" in model_set

    result = train_baseline_models(
        gold_df,
        args.output_dir,
        random_state=args.random_state,
        models=models,
        train_catboost=include_optionals,
        tune_elasticnet=args.tune and ("elasticnet" in model_set or "all" in model_set),
        tune_catboost=args.tune and ("catboost" in model_set or include_optionals),
        tuning_trials=args.tuning_trials,
        run_xgboost=include_optionals,
        run_tabpfn=include_optionals,
        tabpfn_model_paths=args.tabpfn_checkpoint,
        run_autogluon=include_optionals,
        autogluon_time_limit=args.autogluon_time_limit,
    )

    print(
        json.dumps(
            {
                "metrics": result.metrics,
                "output_dir": str(result.output_dir),
                "skipped_models": result.skipped_models,
            },
            indent=2,
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
