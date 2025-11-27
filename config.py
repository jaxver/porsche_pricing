"""
Configuration file for Elferspot Listings project.
Centralizes all paths, constants, and settings.
"""
import os
from pathlib import Path

# Project root directory
PROJECT_ROOT = Path(__file__).parent.absolute()

# Data directories
DATA_DIR = PROJECT_ROOT / "data"
DATA_DIR_RAW = DATA_DIR / "raw"
DATA_DIR_BRONZE = DATA_DIR / "bronze"
DATA_DIR_SILVER = DATA_DIR / "silver"
DATA_DIR_GOLD = DATA_DIR / "gold"

# Output directories
RESULTS_DIR = PROJECT_ROOT / "results"
MODELS_DIR = PROJECT_ROOT / "models"
LOGS_DIR = PROJECT_ROOT / "logs"

# Ensure directories exist
for dir_path in [DATA_DIR, DATA_DIR_RAW, DATA_DIR_BRONZE, DATA_DIR_SILVER, 
                 DATA_DIR_GOLD, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    dir_path.mkdir(parents=True, exist_ok=True)

# Data file paths
EXCHANGE_RATES_FILE = DATA_DIR / "exchange_rates.json"
LISTING_URLS_FILE = DATA_DIR / "listing_urls.csv"
ALL_LISTING_URLS_FILE = DATA_DIR / "all_listing_urls.csv"

# Bronze/Silver/Gold data files
LISTINGS_BRONZE = DATA_DIR_BRONZE / "listings_bronze.xlsx"
LISTINGS_SILVER = DATA_DIR_SILVER / "listings_silver.xlsx"
LISTINGS_GOLD = DATA_DIR_GOLD / "listings_gold.xlsx"

# Model file paths
MODEL_CATBOOST = MODELS_DIR / "catboost_price_model.cbm"
MODEL_RIDGE = MODELS_DIR / "ridge_price_model.pkl"
MODEL_ELASTICNET = MODELS_DIR / "elasticnet_price_model.pkl"

# Feature lists for modeling
NUMERIC_FEATURES = [
    'Mileage_km',
    'Mileage_sq',
    'log_mileage',
    'listing_score',
    'Year of construction'
]

CATEGORICAL_FEATURES = [
    'Series',
    'Model',
    'model_category',
    'Transmission',
    'Drive',
    'Ready to drive',
    'Car location',
    'Matching numbers',
    'Interior color',
    'Paint-to-Sample (PTS)',
    'is_fully_restored',
    'owners_known'
]

# Target variable
TARGET_VARIABLE = 'price_in_eur'

# Scraping configuration
SCRAPING_CONFIG = {
    'base_url': 'https://www.elferspot.com',
    'user_agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
    'request_timeout': 10,
    'max_retries': 3,
    'delay_between_requests': 1.0
}

# Currency configuration
SUPPORTED_CURRENCIES = ['EUR', 'USD', 'GBP', 'JPY', 'CHF']
BASE_CURRENCY = 'EUR'
EXCHANGE_RATE_TTL = 24 * 3600  # 24 hours in seconds

# Model training configuration
MODEL_CONFIG = {
    'test_size': 0.2,
    'random_state': 42,
    'cv_folds': 5,
    'catboost': {
        'iterations': 1000,
        'learning_rate': 0.05,
        'depth': 6,
        'l2_leaf_reg': 3,
        'random_seed': 42,
        'verbose': False
    },
    'ridge': {
        'alphas': [0.1, 1.0, 10.0, 100.0],
        'cv': 5
    },
    'elasticnet': {
        'alpha': 1.0,
        'l1_ratio': 0.5,
        'max_iter': 10000
    }
}

# Data quality thresholds
DATA_QUALITY_CONFIG = {
    'max_price_std_deviations': 3,
    'max_mileage_std_deviations': 3,
    'min_price_eur': 5000,
    'max_price_eur': 5000000,
    'min_year': 1950,
    'max_year': 2026
}

# Logging configuration
LOG_FORMAT = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
LOG_DATE_FORMAT = '%Y-%m-%d %H:%M:%S'
LOG_LEVEL = 'INFO'
