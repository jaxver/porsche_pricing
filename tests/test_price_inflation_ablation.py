from __future__ import annotations

import pandas as pd

from elferspot_listings.modeling.price_inflation_ablation import bootstrap_mean_ci, run_price_inflation_ablation


def test_bootstrap_mean_ci_raises_on_empty_input():
    try:
        bootstrap_mean_ci([])
    except ValueError as exc:
        assert "delta" in str(exc)
    else:
        raise AssertionError("bootstrap_mean_ci should reject empty input")


def test_run_price_inflation_ablation_reports_paired_delta_and_ci():
    df = pd.DataFrame({"price_inflation_factor": [1.0, 1.1], "price_in_eur": [100000.0, 110000.0]})

    def fake_score_fn(frame, seed, model_name, use_feature):
        base = 100.0 + seed
        return base if use_feature else base + 10.0

    result = run_price_inflation_ablation(
        df,
        seeds=[1, 2, 3],
        model_name="ridge",
        bootstrap_iterations=100,
        score_fn=fake_score_fn,
    )

    assert result["model_name"] == "ridge"
    assert result["seeds"] == [1, 2, 3]
    assert result["delta_mae_per_seed"] == [10.0, 10.0, 10.0]
    assert result["delta_mae_mean"] == 10.0
    assert result["delta_mae_ci"] == [10.0, 10.0]
