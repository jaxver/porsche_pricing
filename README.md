# Elferspot Listings - Porsche Price Analytics

A professional, production-ready Python application for analyzing Porsche car listings with advanced machine learning capabilities for price prediction and market analysis. Features a complete ETL pipeline (Bronze → Silver → Gold), multiple ML models, and an interactive dashboard.

## 🚗 Features

- **Structured Data Pipeline**: Bronze → Silver → Gold medallion architecture
- **Modular Code Organization**: Reusable Python modules for data processing and modeling
- **Machine Learning Models**: CatBoost, Ridge, and ElasticNet regression models
- **Interactive Dashboard**: Streamlit web application for browsing and analyzing listings
- **Automated Data Processing**: Standardized cleaning, feature engineering, and validation
- **Price Analytics**: Identify underpriced listings and market trends
- **Currency Conversion**: Automatic EUR conversion with cached exchange rates
- **Comprehensive Notebooks**: Step-by-step workflow from data gathering to modeling

## 🏗️ Project Structure

```
elferspot_prod/
├── config.py                     # Central configuration (paths, settings)
├── requirements.txt              # Python dependencies
│
├── elferspot_listings/           # Main Python package
│   ├── __init__.py
│   ├── data_processing/          # Data transformation modules
│   │   ├── bronze_to_silver.py   # Raw → Clean data
│   │   └── silver_to_gold.py     # Clean → Model-ready data
│   ├── models/                   # ML model modules (future)
│   └── utils/                    # Utility functions
│       ├── exchange_rates.py     # Currency conversion
│       └── helpers.py            # Data loading, logging, etc.
│
├── notebooks/                    # Analysis notebooks (organized by stage)
│   ├── 01_data_gathering/        # Web scraping
│   │   └── 02_scraping_notebook.ipynb
│   ├── 02_data_preparation/      # Data cleaning
│   │   ├── 01_listings_bronzetosilver.ipynb
│   │   └── 02_bronze_to_silver.ipynb
│   ├── 03_feature_engineering/   # Feature creation
│   │   └── 03_silver_to_gold.ipynb
│   ├── 04_modeling/              # Model training
│   │   ├── 04_catboost_model.ipynb
│   │   ├── 05_ridge_regression.ipynb
│   │   └── 06_elasticnet_regression.ipynb
│   └── 05_analysis/              # Exploratory analysis
│       ├── 01_predictive_regression.ipynb
│       ├── 02_vif_diagnostics.ipynb
│       └── 03_market_insights.ipynb
│
├── app/
│   └── streamlit_app.py          # Interactive dashboard
│
├── data/                         # Data storage (gitignored)
│   ├── raw/                      # Original scraped data
│   ├── bronze/                   # Raw data layer
│   ├── silver/                   # Cleaned data layer
│   └── gold/                     # Model-ready data layer
│
├── models/                       # Trained models (gitignored)
├── results/                      # Analysis outputs (gitignored)
├── logs/                         # Log files (gitignored)
└── tests/                        # Unit tests
    └── test_basic.py
```


## 📊 Data Pipeline

The project follows a **medallion architecture** with three data quality layers:

### 1. Bronze Layer (Raw Data)
- **Location:** `data/bronze/`
- **Content:** Raw scraped data with minimal processing
- **Notebook:** `01_data_gathering/02_scraping_notebook.ipynb`

### 2. Silver Layer (Cleaned Data)
- **Location:** `data/silver/`
- **Content:** Cleaned, validated, and standardized data
- **Processing:** 
  - Remove duplicates
  - Standardize formats
  - Convert currencies to EUR
  - Clean and validate fields
- **Module:** `elferspot_listings.data_processing.bronze_to_silver`
- **Notebook:** `02_data_preparation/02_bronze_to_silver.ipynb`

### 3. Gold Layer (Model-Ready Data)
- **Location:** `data/gold/`
- **Content:** Feature-engineered data ready for ML models
- **Processing:**
  - Remove outliers
  - Create log-transformed features
  - Engineer derived features (listing scores, model categories)
  - Prepare modeling datasets
- **Module:** `elferspot_listings.data_processing.silver_to_gold`
- **Notebook:** `03_feature_engineering/03_silver_to_gold.ipynb`

## 🚀 Getting Started

### Prerequisites

- Python 3.8 or higher
- pip package manager

### Installation

1. Clone the repository:
```bash
git clone https://github.com/jaxver/Elferspot-Scraper.git
cd Elferspot-Scraper
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

3. Configure paths (optional):
   - Review `config.py` for data paths and settings
   - Adjust as needed for your environment

### Quick Start Workflow

#### Option 1: Using Python Modules (Recommended for Production)

```python
from pathlib import Path
from config import LISTINGS_BRONZE, LISTINGS_SILVER, LISTINGS_GOLD
from elferspot_listings.data_processing import process_bronze_to_silver, process_silver_to_gold

# Process Bronze → Silver
process_bronze_to_silver(LISTINGS_BRONZE, LISTINGS_SILVER)

# Process Silver → Gold
process_silver_to_gold(LISTINGS_SILVER, LISTINGS_GOLD)
```

#### Option 2: Using Jupyter Notebooks (Recommended for Learning/Development)

Run notebooks in order:

1. **Data Gathering:**
   ```
   notebooks/01_data_gathering/02_scraping_notebook.ipynb
   ```

2. **Data Preparation:**
   ```
   notebooks/02_data_preparation/02_bronze_to_silver.ipynb
   ```

3. **Feature Engineering:**
   ```
   notebooks/03_feature_engineering/03_silver_to_gold.ipynb
   ```

4. **Model Training:**
   ```
   notebooks/04_modeling/04_catboost_model.ipynb
   notebooks/04_modeling/05_ridge_regression.ipynb
   notebooks/04_modeling/06_elasticnet_regression.ipynb
   ```

5. **Analysis:**
   ```
   notebooks/05_analysis/03_market_insights.ipynb
   ```

### Running the Dashboard

Launch the Streamlit application:
```bash
streamlit run app/streamlit_app.py
```

The dashboard provides:
- Interactive filtering by model, year, mileage, price
- Time series visualizations
- Model-based analytics and comparisons
- Direct links to individual listings

## 🤖 Machine Learning Models

The project implements multiple regression approaches:

### CatBoost (Primary Model)
- **Notebook:** `notebooks/04_modeling/04_catboost_model.ipynb`
- **Features:** Handles categorical variables natively, gradient boosting
- **Performance:** Typically achieves R² > 0.85
- **Use Case:** Production price predictions

### Ridge Regression
- **Notebook:** `notebooks/04_modeling/05_ridge_regression.ipynb`
- **Features:** Linear model with L2 regularization
- **Use Case:** Baseline comparison, interpretable coefficients

### ElasticNet
- **Notebook:** `notebooks/04_modeling/06_elasticnet_regression.ipynb`
- **Features:** Combined L1/L2 regularization
- **Use Case:** Feature selection, sparse models

Models are evaluated using:
- **Cross-validation:** 5-fold CV for robust performance estimates
- **Metrics:** RMSE, MAE, R²
- **Feature Importance:** For interpretability

Each modeling notebook saves calibrated prediction intervals to `results/model_predictions/`, enabling downstream analysis notebooks (`05_analysis/`) to compare models and flag under/over-valued listings without re-training.*** End Patch

## 🛠️ Development

### Project Configuration

All configuration is centralized in `config.py`:
- Data paths (Bronze/Silver/Gold layers)
- Model hyperparameters
- Feature definitions
- Scraping settings

### Running Tests
```bash
pytest tests/
```

### Code Organization Principles

1. **Separation of Concerns:** Data processing, modeling, and analysis are separate
2. **Reusability:** Common functions in utility modules
3. **Reproducibility:** Fixed random seeds, versioned data outputs
4. **Modularity:** Each notebook/module has a single, clear purpose

### Adding New Features

1. **Data Processing:** Add functions to `elferspot_listings/data_processing/`
2. **Utilities:** Add to `elferspot_listings/utils/`
3. **Models:** Add to `elferspot_listings/models/`
4. **Analysis:** Create new notebooks in appropriate `notebooks/` subdirectory

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- Data sourced from public Porsche listing platforms
- Built with Python, pandas, scikit-learn, CatBoost, and Streamlit

## 📧 Contact

**Jaxver** - [@jaxver](https://github.com/jaxver)

For questions or collaboration opportunities, please open an issue on GitHub.

---

**Note**: This project is for educational and research purposes. Always respect website terms of service when scraping data.