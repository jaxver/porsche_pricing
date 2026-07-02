# Elferspot Listings - Porsche Price Analytics

An end-to-end Porsche listing price analytics project that demonstrates reproducible data pipelines, feature engineering, tabular ML baselines, CatBoost benchmarking, optional TabPFN/TabFM/AutoGluon comparisons, model artifacts, and a Streamlit dashboard.

## What This Repo Shows

- Bronze -> Silver -> Gold data flow for repeatable preparation
- Curated notebooks that tell the analysis story in order
- Default tabular baselines plus CatBoost benchmarking
- Optional advanced challenger runs for TabPFN, TabFM, and AutoGluon
- Saved model artifacts and benchmark outputs for later review
- A Streamlit dashboard for browsing listings and results

## Project Layout

```text
elferspot_prod/
├── app/streamlit_app.py          # Streamlit dashboard
├── data/                         # Bronze/Silver/Gold inputs and outputs (gitignored)
├── models/                       # Saved model artifacts (gitignored)
├── results/                      # Benchmark and analysis outputs (gitignored)
├── logs/                         # Runtime logs (gitignored)
├── catboost_info/                # CatBoost training output (gitignored)
├── pyproject.toml                # Project metadata and dependency extras
├── notebooks/                    # Curated notebook workflow
└── tests/                        # Minimal smoke tests
```

## Setup

Default install and smoke checks:

```powershell
python -m pip install -e .
pytest
streamlit run app/streamlit_app.py
```

## Advanced Benchmark Setup

For the optional TabPFN, TabFM, and AutoGluon comparison path:

```powershell
python -m pip install -e ".[advanced]"
```

TabPFN and TabFM may download weights or checkpoints the first time they are used. AutoGluon can create large local benchmark artifacts, so keep the advanced extras separate from the default install and only run them when you need the full challenger comparison.

## Notebook Workflow

Use the curated sequence below for the main portfolio path. This is the recommended portfolio workflow and supersedes older research-notebook handoff text that points at `03_silver_to_gold.ipynb`.

1. `notebooks/02_data_preparation/01_listings_bronzetosilver.ipynb`
2. `notebooks/02_data_preparation/02_bronze_to_silver.ipynb`
3. `notebooks/03_feature_engineering/01_silver_to_gold_walkthrough.ipynb`
4. `notebooks/04_modeling/01_baseline_sklearn_skrub.ipynb`
5. `notebooks/04_modeling/02_catboost_benchmark.ipynb`
6. `notebooks/04_modeling/03_cutting_edge_challengers.ipynb`
7. `notebooks/04_modeling/04_model_comparison_report.ipynb`
8. `notebooks/05_analysis/01_market_insights.ipynb`

If you need to regenerate Bronze data first, start with the scraping notebook and then continue with the sequence above.

## Bronze, Silver, Gold

- Bronze: raw scraped listings
- Silver: cleaned and standardized records
- Gold: feature-engineered, model-ready data

Each stage is designed to be reproducible and easy to rerun without rewriting downstream analysis by hand.

## Benchmark Outputs

Benchmark runs write metrics, predictions, and model artifacts into `results/benchmarks/<run_id>/`. Those outputs are meant for local analysis and dashboard review, not for version control.

## Generated Output Policy

Keep `data/`, `results/`, `logs/`, `models/`, `catboost_info/`, and generated benchmark artifacts uncommitted unless a specific sanitized artifact is intentionally published.

Notebook outputs are stripped by default before commit. This protects local paths such as `C:\Users\...`, `\\NAS_...`, `/Users/...`, `/home/...`, temp directories, tracebacks, and rich-output metadata from leaking into git history.

Use the notebook hygiene check before publishing changes:

```powershell
python scripts/check_notebook_hygiene.py notebooks
```

Clear notebook outputs when needed:

```powershell
jupyter nbconvert --clear-output --inplace notebooks/path/to/notebook.ipynb
nbstripout notebooks/path/to/notebook.ipynb
```

If a demo output is intentionally worth keeping, set that cell metadata to `{"keep_output": true}` and manually review it for private paths or data before committing. Prefer curated screenshots or exported summaries in docs over committing arbitrary executed notebook output.

## Dashboard

Launch the Streamlit app with:

```powershell
streamlit run app/streamlit_app.py
```

The dashboard expects `data/all_listings_gold.xlsx` or an equivalent Gold export to exist before launch. It is intended to present the cleaned data, benchmark outputs, and model comparisons without training anything on import.

## License

MIT
