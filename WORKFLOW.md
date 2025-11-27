# Elferspot Listings - Workflow Guide

## 📋 Complete Pipeline Workflow

### Stage 1: Data Gathering 🔍
**Goal:** Scrape Porsche listings from web sources

**Notebook:** `notebooks/01_data_gathering/02_scraping_notebook.ipynb`

**Process:**
1. Configure scraping parameters
2. Fetch listing URLs
3. Parse HTML content
4. Extract structured data
5. Save to Bronze layer

**Output:** `data/bronze/listings_bronze_YYYYMMDD_HHMMSS.xlsx`

---

### Stage 2: Data Preparation 🧹
**Goal:** Clean and standardize raw data

**Notebook:** `notebooks/02_data_preparation/02_bronze_to_silver.ipynb`

**Module:** `elferspot_listings.data_processing.bronze_to_silver`

**Process:**
1. Load bronze data
2. Remove duplicates
3. Clean mileage values (convert miles to km)
4. Standardize condition fields
5. Convert all prices to EUR
6. Create derived features (owners_known, is_fully_restored)
7. Validate data quality

**Output:** `data/silver/listings_silver_YYYYMMDD_HHMMSS.xlsx`

---

### Stage 3: Feature Engineering 🔧
**Goal:** Create model-ready features

**Notebook:** `notebooks/03_feature_engineering/03_silver_to_gold.ipynb`

**Module:** `elferspot_listings.data_processing.silver_to_gold`

**Process:**
1. Load silver data
2. Remove price outliers (±3σ from log-price mean)
3. Create log-transformed features (log_price, log_mileage)
4. Create polynomial features (Mileage_sq)
5. Categorize models into broader groups
6. Calculate listing quality scores
7. Prepare final feature matrix

**Output:** `data/gold/listings_gold_YYYYMMDD_HHMMSS.xlsx`

---

### Stage 4: Model Training 🤖
**Goal:** Train price prediction models

**Notebooks:**
- `notebooks/04_modeling/04_catboost_model.ipynb` - CatBoost (primary)
- `notebooks/04_modeling/05_ridge_model.ipynb` - Ridge regression
- `notebooks/04_modeling/06_model_comparison.ipynb` - Compare all models

**Process:**
1. Load gold data
2. Define feature sets (numeric + categorical)
3. Train-test split (80/20)
4. Train model with cross-validation
5. Evaluate performance (RMSE, MAE, R²)
6. Analyze feature importance
7. Save trained model

**Outputs:**
- `models/catboost_model_YYYYMMDD_HHMMSS.cbm`
- `models/feature_importance_YYYYMMDD_HHMMSS.csv`
- `models/model_metrics_YYYYMMDD_HHMMSS.csv`

---

### Stage 5: Analysis & Insights 📊
**Goal:** Generate business insights

**Notebook:** `notebooks/05_analysis/07_price_analysis.ipynb`

**Process:**
1. Load gold data and trained models
2. Generate price predictions
3. Identify underpriced listings
4. Analyze market trends
5. Create visualizations
6. Export results

**Outputs:** `results/underpriced_listings.xlsx`

---

## 🎯 Quick Start Commands

### Full Pipeline (Python)
```python
from config import LISTINGS_BRONZE, LISTINGS_SILVER, LISTINGS_GOLD
from elferspot_listings.data_processing import process_bronze_to_silver, process_silver_to_gold

# Run complete pipeline
process_bronze_to_silver(LISTINGS_BRONZE, LISTINGS_SILVER)
process_silver_to_gold(LISTINGS_SILVER, LISTINGS_GOLD)
```

### Dashboard
```bash
streamlit run app/streamlit_app.py
```

---

## 🔄 Data Flow Diagram

```
[Web Sources]
     ↓
[Scraping Script] → data/bronze/ (Raw data)
     ↓
[Bronze → Silver] → data/silver/ (Clean data)
     ↓
[Silver → Gold]   → data/gold/ (Model-ready)
     ↓
[Model Training]  → models/ (Trained models)
     ↓
[Analysis]        → results/ (Insights)
     ↓
[Dashboard]       → [Interactive UI]
```

---

## 📂 File Naming Conventions

- **Bronze:** `listings_bronze_YYYYMMDD_HHMMSS.xlsx`
- **Silver:** `listings_silver_YYYYMMDD_HHMMSS.xlsx`
- **Gold:** `listings_gold_YYYYMMDD_HHMMSS.xlsx`
- **Models:** `{model_type}_model_YYYYMMDD_HHMMSS.{ext}`
- **Results:** `{analysis_type}_YYYYMMDD_HHMMSS.{ext}`

All files are timestamped for version tracking.

---

## 🛠️ Configuration

All settings in `config.py`:
- Data paths
- Model hyperparameters
- Feature definitions
- Data quality thresholds

---

## ✅ Best Practices

1. **Always start from Bronze** - Reprocess from raw data when changing cleaning logic
2. **Version your data** - Timestamp files allow tracking pipeline runs
3. **Log transformations** - Each notebook logs operations for reproducibility
4. **Validate at each stage** - Check data quality before moving to next stage
5. **Save intermediate results** - Don't recompute expensive operations
6. **Document changes** - Update this workflow guide when modifying pipeline

---

## 🐛 Troubleshooting

### Issue: "Bronze file not found"
**Solution:** Run `02_scraping_notebook.ipynb` first, or place existing data in `data/bronze/`

### Issue: "Missing columns in Silver"
**Solution:** Check Bronze data structure matches expected schema

### Issue: "Model training fails"
**Solution:** Ensure Gold data has required features and no missing target values

### Issue: "Currency conversion errors"
**Solution:** Check internet connection for exchange rate API, or verify cached rates in `data/exchange_rates.json`

---

For more details, see the main [README.md](README.md)
