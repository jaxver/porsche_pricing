from __future__ import annotations

import argparse
import json
from pathlib import Path

import config
import pandas as pd

from .train import train_baseline_models


MODEL_CHOICES = ["median", "ridge", "elasticnet", "skrub_ridge", "xgboost", "perpetual", "catboost", "tabpfn", "tabfm", "autogluon", "all"]


def _parse_autogluon_dynamic_stacking(value: str) -> bool | None:
    if value == "auto":
        return None
    if value == "true":
        return True
    if value == "false":
        return False
    raise ValueError(f"Unsupported AutoGluon dynamic stacking value: {value}")


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
    parser.add_argument("--device", choices=("cpu", "gpu"), default="cpu", help="Execution device for benchmark models.")
    parser.add_argument("--gpu-devices", default=None, help="CatBoost GPU device IDs such as 0 or 0:1.")
    parser.add_argument("--tune", action="store_true", help="Tune ElasticNet and CatBoost when those models are selected.")
    parser.add_argument("--tuning-trials", type=int, default=25, help="Number of Optuna trials for tuning.")
    parser.add_argument(
        "--tabpfn-checkpoint",
        action="append",
        default=None,
        help="TabPFN checkpoint alias or .ckpt path; repeat to run multiple checkpoints.",
    )
    parser.add_argument(
        "--tabpfn-backend",
        choices=("local", "client"),
        default="local",
        help="TabPFN backend to use.",
    )
    parser.add_argument("--tabpfn-thinking", action="store_true", help="Enable TabPFN client thinking mode.")
    parser.add_argument(
        "--tabpfn-thinking-effort",
        choices=("medium", "high"),
        default="medium",
        help="Thinking effort for the TabPFN client backend.",
    )
    parser.add_argument(
        "--tabpfn-thinking-timeout",
        type=float,
        default=None,
        help="Optional thinking timeout in seconds for the TabPFN client backend.",
    )
    parser.add_argument(
        "--tabpfn-thinking-metric",
        choices=("rmse", "mae"),
        default="rmse",
        help="Thinking metric for the TabPFN client backend.",
    )
    parser.add_argument("--autogluon-time-limit", type=int, default=600, help="Time limit in seconds for AutoGluon.")
    parser.add_argument("--autogluon-presets", default="best_quality", help="AutoGluon presets passed to fit.")
    parser.add_argument(
        "--autogluon-dynamic-stacking",
        choices=("auto", "true", "false"),
        default="auto",
        help="AutoGluon dynamic stacking mode.",
    )
    parser.add_argument("--autogluon-clean-output", action="store_true", help="Remove the AutoGluon artifact dir before fitting.")
    parser.add_argument("--include-optionals", action="store_true", help="Run optional models when used together with --model all.")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = _build_parser()
    args = parser.parse_args(argv)
    models = args.model or ["all"]
    model_set = set(models)
    include_optionals = args.include_optionals and "all" in model_set
    tabpfn_requested = "tabpfn" in model_set or ("all" in model_set and (include_optionals or args.tabpfn_checkpoint is not None))

    if tabpfn_requested and args.tabpfn_thinking and args.tabpfn_backend != "client":
        parser.error("--tabpfn-thinking requires --tabpfn-backend client.")
    if tabpfn_requested and args.tabpfn_backend == "client" and args.tabpfn_checkpoint is not None:
        parser.error("--tabpfn-checkpoint is local-backend only; remove it when using --tabpfn-backend client.")

    gold_df = pd.read_excel(args.input)

    train_kwargs = {
        "random_state": args.random_state,
        "models": models,
        "train_catboost": include_optionals,
        "tune_elasticnet": args.tune and ("elasticnet" in model_set or "all" in model_set),
        "tune_catboost": args.tune and ("catboost" in model_set or include_optionals),
        "tuning_trials": args.tuning_trials,
        "run_xgboost": include_optionals,
        "run_perpetual": include_optionals,
        "run_tabpfn": include_optionals,
        "run_tabfm": include_optionals,
        "tabpfn_model_paths": args.tabpfn_checkpoint,
        "tabpfn_backend": args.tabpfn_backend,
        "tabpfn_thinking": args.tabpfn_thinking,
        "tabpfn_thinking_effort": args.tabpfn_thinking_effort,
        "tabpfn_thinking_timeout": args.tabpfn_thinking_timeout,
        "tabpfn_thinking_metric": args.tabpfn_thinking_metric,
        "run_autogluon": include_optionals,
        "autogluon_time_limit": args.autogluon_time_limit,
        "autogluon_presets": args.autogluon_presets,
        "autogluon_dynamic_stacking": _parse_autogluon_dynamic_stacking(args.autogluon_dynamic_stacking),
        "autogluon_clean_output": args.autogluon_clean_output,
    }
    if args.device == "gpu" or args.gpu_devices is not None:
        train_kwargs["device"] = args.device
        train_kwargs["gpu_devices"] = args.gpu_devices

    result = train_baseline_models(gold_df, args.output_dir, **train_kwargs)

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
