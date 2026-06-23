# Elferspot Listings - Workflow Guide

## Overview

The main project flow is Bronze -> Silver -> Gold, followed by curated notebooks for baseline modeling, CatBoost benchmarking, optional challenger comparisons, and market analysis.

## Default Path

Use the default install for the standard portfolio path:

```powershell
python -m pip install -r requirements.txt
```

Then run the notebook sequence below.

## Curated Notebook Sequence

This curated sequence is the recommended portfolio path and supersedes older research-notebook handoff text that points at `03_silver_to_gold.ipynb`.

1. `notebooks/02_data_preparation/01_listings_bronzetosilver.ipynb`
2. `notebooks/02_data_preparation/02_bronze_to_silver.ipynb`
3. `notebooks/03_feature_engineering/01_silver_to_gold_walkthrough.ipynb`
4. `notebooks/04_modeling/01_baseline_sklearn_skrub.ipynb`
5. `notebooks/04_modeling/02_catboost_benchmark.ipynb`
6. `notebooks/04_modeling/04_model_comparison_report.ipynb`
7. `notebooks/05_analysis/01_market_insights.ipynb`

If you are refreshing raw data first, start with the scraping notebook and then continue with the curated sequence.

## Bronze -> Silver -> Gold

- Bronze: scrape or collect raw listings
- Silver: clean, standardize, and validate the raw export
- Gold: engineer features for modeling and comparison

Each stage should be rerunnable without manual copying between notebooks.

## Benchmark Runner Outputs

Default benchmark runs should write into `results/benchmarks/<run_id>/`, including metrics, predictions, and any saved artifacts needed for review. Keep those outputs local unless a sanitized artifact is intentionally published.

## Advanced Benchmark Path

For the optional challenger comparison path:

```powershell
python -m pip install -r requirements-advanced.txt
```

Use the advanced notebook only when you need TabPFN or AutoGluon comparisons. TabPFN may download checkpoints on first use, and AutoGluon can generate large local benchmark artifacts.

## Dashboard Handoff

The Streamlit app should read benchmark outputs when they exist and stay usable when they do not:

```powershell
streamlit run app/streamlit_app.py
```

The app expects `data/all_listings_gold.xlsx` or an equivalent Gold export to exist before launch. The app is presentation-only; it should not perform training on import.

## Generated Output Policy

Keep `data/`, `results/`, `logs/`, `models/`, `catboost_info/`, and generated benchmark artifacts out of git unless a specific sanitized artifact is intentionally published.

## Notes

- Use `pytest` for local smoke checks.
- If `pytest` is intercepted in this environment, use `rtk proxy pytest`.
- Keep notebook logic focused on narrative and analysis, not duplicated pipeline code.
