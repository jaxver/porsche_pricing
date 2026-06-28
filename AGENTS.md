# AGENTS.md — Porsche Listings Analysis

## Quick start

```powershell
python -m pip install -e .
```

## Project structure

- `data/` — Bronze → Silver → Gold Excel pipeline. All files binary `.xlsx`. Gitignored.
- `elferspot_listings/` — empty namespace package. Real code lives in notebooks.
- `elferspot_listings/notebooks/` — **production** notebooks (bronze→silver pipeline, listing scores).
- `notebooks/` — **research** notebooks (CatBoost, Ridge, ElasticNet, VIF, scraping).
- `app/streamlit_app.py` — Streamlit dashboard: `streamlit run app/streamlit_app.py`
- `scripts/exchange_rates.py` — Currency rate fetcher with 24h cache + static fallback.
- `tests/test_basic.py` — Minimal import test.

## Data pipeline

1. **Bronze** → `clean_excel_file()` in `bronze_to_silver_production.ipynb` → **Silver**
2. Silver → feature engineering in research notebooks → **Gold**
3. Source: `data/all_listings_bronze.xlsx` → `all_listings_silver.xlsx` → `all_listings_gold.xlsx`

## Model notes (CatBoost)

- Target: `log(price_in_eur)` — log-transformed for stability.
- Loss: `Quantile:alpha=0.05/0.5/0.95` for confidence intervals (3 separate models).
- GPU by default: `task_type: 'GPU'` — falls back to CPU if unavailable.
- Categorical features passed by column index to `Pool(cat_features=...)`.
- Hyperparameters tuned with Optuna in `notebooks/Catboost model.ipynb`.
- Feature categories (ordered ordinal) used in interaction terms (e.g., `Mileage_km * model_cat_ordered`).

## Conventions & quirks

- Price filter: 15,000 < price_in_eur < 700,000.
- Mileage: extracted from text field `"Mileage"` via regex, converted mi→km (×1.60934).
- Currency: hardcoded static fallback rates in notebooks AND `scripts/exchange_rates.py`. Keep in sync.
- Listing score: weighted sum of description-derived regex flags (weights in notebooks).
- All notebooks assume paths relative to their own location (`../data/`).
- `data/`, `results/`, `Archive/`, `changelogs/`, `catboost_info/` are gitignored — do not commit.
