from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Callable, Sequence

import numpy as np
import pandas as pd

import config

from .train import train_baseline_models


DEFAULT_SEEDS = (42, 43, 44, 45, 46)
DEFAULT_OUTPUT = Path(config.RESULTS_DIR) / "benchmarks" / "price_inflation_ablation.json"


def bootstrap_mean_ci(values: Sequence[float], *, iterations: int = 2000, seed: int = 42, alpha: float = 0.05) -> tuple[float, float]:
    samples = np.asarray(values, dtype=float)
    if samples.size == 0:
        raise ValueError("At least one delta is required for bootstrap CI")
    rng = np.random.default_rng(seed)
    boot_means = np.array([
        rng.choice(samples, size=samples.size, replace=True).mean()
        for _ in range(iterations)
    ])
    lower = float(np.percentile(boot_means, 100 * (alpha / 2)))
    upper = float(np.percentile(boot_means, 100 * (1 - alpha / 2)))
    return lower, upper


def _score_variant(
    df: pd.DataFrame,
    *,
    seed: int,
    model_name: str,
    use_inflation_feature: bool,
) -> float:
    variant = df.copy()
    if not use_inflation_feature:
        variant = variant.drop(columns=["price_inflation_factor"], errors="ignore")

    result = train_baseline_models(
        variant,
        output_dir=Path(config.RESULTS_DIR) / "benchmarks" / f"price_inflation_ablation_{model_name}_{seed}_{'with' if use_inflation_feature else 'without'}",
        random_state=seed,
        models=[model_name],
        train_catboost=False,
        run_xgboost=False,
        run_perpetual=False,
        run_tabpfn=False,
        run_tabfm=False,
        run_autogluon=False,
        verbose=False,
    )
    return float(result.metrics[model_name]["mae_eur"])


def run_price_inflation_ablation(
    df: pd.DataFrame,
    *,
    seeds: Sequence[int] = DEFAULT_SEEDS,
    model_name: str = "ridge",
    bootstrap_iterations: int = 2000,
    score_fn: Callable[[pd.DataFrame, int, str, bool], float] | None = None,
) -> dict[str, object]:
    scorer = score_fn or (lambda frame, seed, name, use_feature: _score_variant(frame, seed=seed, model_name=name, use_inflation_feature=use_feature))

    with_feature: list[float] = []
    without_feature: list[float] = []

    for seed in seeds:
        with_feature.append(float(scorer(df, seed, model_name, True)))
        without_feature.append(float(scorer(df, seed, model_name, False)))

    deltas = [without - with_ for with_, without in zip(with_feature, without_feature)]
    ci_low, ci_high = bootstrap_mean_ci(deltas, iterations=bootstrap_iterations)

    return {
        "model_name": model_name,
        "seeds": list(seeds),
        "with_feature_mae": with_feature,
        "without_feature_mae": without_feature,
        "delta_mae_per_seed": deltas,
        "delta_mae_mean": float(np.mean(deltas)),
        "delta_mae_ci": [ci_low, ci_high],
    }


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Run a price-inflation ablation and report holdout MAE deltas.")
    parser.add_argument("--input", type=Path, default=Path(config.LISTINGS_GOLD), help="Gold-layer dataset to evaluate.")
    parser.add_argument("--model", default="ridge", help="Model to score (default: ridge).")
    parser.add_argument("--seed", action="append", type=int, default=None, help="Repeated random seed; repeat for multiple seeds.")
    parser.add_argument("--bootstrap-iterations", type=int, default=2000, help="Bootstrap resamples for the confidence interval.")
    parser.add_argument("--output-json", type=Path, default=DEFAULT_OUTPUT, help="Path to write the JSON summary.")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    seeds = args.seed or list(DEFAULT_SEEDS)
    df = pd.read_excel(args.input)
    summary = run_price_inflation_ablation(
        df,
        seeds=seeds,
        model_name=args.model,
        bootstrap_iterations=args.bootstrap_iterations,
    )
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(json.dumps(summary, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
