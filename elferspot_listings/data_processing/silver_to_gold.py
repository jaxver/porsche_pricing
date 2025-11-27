"""
Silver to Gold data transformation.
Applies feature engineering and prepares data for modeling.
"""
import pandas as pd
import numpy as np
import logging
from pathlib import Path
from typing import Optional, List
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

logger = logging.getLogger(__name__)


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    n_std: float = 3.0,
    use_log: bool = False
) -> pd.DataFrame:
    """Remove outliers based on standard deviations."""
    logger.info(f"Removing outliers from {column} (±{n_std} std)")
    
    initial_count = len(df)
    
    if use_log:
        log_col = f'log_{column}'
        df[log_col] = np.log(df[column])
        mean_val = df[log_col].mean()
        std_val = df[log_col].std()
        df = df[
            (df[log_col] >= mean_val - n_std * std_val) & 
            (df[log_col] <= mean_val + n_std * std_val)
        ].copy()
        df = df.drop(columns=[log_col])
    else:
        mean_val = df[column].mean()
        std_val = df[column].std()
        df = df[
            (df[column] >= mean_val - n_std * std_val) & 
            (df[column] <= mean_val + n_std * std_val)
        ].copy()
    
    removed = initial_count - len(df)
    logger.info(f"Removed {removed} outliers ({removed/initial_count*100:.1f}%)")
    
    return df


def create_log_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create log-transformed features."""
    logger.info("Creating log-transformed features")
    
    if 'price_in_eur' in df.columns:
        df['log_price'] = np.log(df['price_in_eur'])
    
    if 'Mileage_km' in df.columns:
        df['log_mileage'] = np.log1p(df['Mileage_km'])
        df['Mileage_sq'] = df['Mileage_km'] ** 2
    
    return df


def create_model_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create broader model categories for better generalization."""
    logger.info("Creating model categories")
    
    if 'Model' not in df.columns:
        logger.warning("Model column not found, skipping categorization")
        return df
    
    # Example categorization - adjust based on your data
    model_mapping = {
        '911': '911',
        '912': '912',
        '914': '914',
        '924': '924',
        '928': '928',
        '944': '944',
        '968': '968',
        'Boxster': 'Boxster',
        'Cayman': 'Cayman',
        'Carrera GT': 'Supercar',
        '918': 'Supercar',
        'Cayenne': 'SUV',
        'Macan': 'SUV',
        'Panamera': 'Sedan',
        'Taycan': 'Electric',
    }
    
    def categorize_model(model: str) -> str:
        if pd.isna(model):
            return 'Other'
        model = str(model)
        for key, category in model_mapping.items():
            if key.lower() in model.lower():
                return category
        return 'Other'
    
    df['model_category'] = df['Model'].apply(categorize_model)
    
    logger.info(f"Created {df['model_category'].nunique()} model categories")
    return df


def calculate_listing_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a quality score for each listing based on completeness.
    Higher score = more complete/quality listing data.
    """
    logger.info("Calculating listing quality scores")
    
    score_components = []
    
    # Check for presence of key fields
    if 'Matching numbers' in df.columns:
        score_components.append(
            (df['Matching numbers'] != 'Unknown').astype(int) * 10
        )
    
    if 'Number of vehicle owners' in df.columns:
        score_components.append(
            (df['Number of vehicle owners'] != 'Unknown').astype(int) * 10
        )
    
    if 'Interior color' in df.columns:
        score_components.append(df['Interior color'].notna().astype(int) * 5)
    
    if 'Exterior color' in df.columns:
        score_components.append(df['Exterior color'].notna().astype(int) * 5)
    
    if 'Paint-to-Sample (PTS)' in df.columns:
        score_components.append(df['Paint-to-Sample (PTS)'].astype(int) * 15)
    
    if 'is_fully_restored' in df.columns:
        score_components.append(df['is_fully_restored'] * 20)
    
    # Low mileage bonus
    if 'Mileage_km' in df.columns:
        mileage_score = np.where(df['Mileage_km'] < 50000, 10, 0)
        score_components.append(pd.Series(mileage_score, index=df.index))
    
    # Combine scores
    if score_components:
        df['listing_score'] = pd.concat(score_components, axis=1).sum(axis=1)
    else:
        df['listing_score'] = 0
    
    logger.info(f"Listing scores range: {df['listing_score'].min():.0f} - {df['listing_score'].max():.0f}")
    
    return df


def prepare_modeling_features(df: pd.DataFrame) -> pd.DataFrame:
    """Prepare features specifically for modeling."""
    logger.info("Preparing modeling features")
    
    # Ensure numeric types
    numeric_cols = ['Mileage_km', 'price_in_eur', 'Year of construction']
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Fill missing values strategically
    if 'Interior color' in df.columns:
        df['Interior color'] = df['Interior color'].fillna('Unknown')
    
    if 'Exterior color' in df.columns:
        df['Exterior color'] = df['Exterior color'].fillna('Unknown')
    
    return df


def process_silver_to_gold(
    silver_path: Path,
    gold_path: Path,
    remove_price_outliers: bool = True,
    outlier_std: float = 3.0
) -> pd.DataFrame:
    """
    Main function to process Silver data to Gold.
    
    Args:
        silver_path: Path to silver (cleaned) data file
        gold_path: Path to save gold (model-ready) data
        remove_price_outliers: Whether to remove price outliers
        outlier_std: Number of standard deviations for outlier removal
    
    Returns:
        Model-ready DataFrame
    """
    logger.info(f"Starting Silver -> Gold transformation")
    logger.info(f"Input: {silver_path}")
    logger.info(f"Output: {gold_path}")
    
    # Load silver data
    df = pd.read_excel(silver_path)
    logger.info(f"Loaded {len(df)} rows from silver")
    
    # Remove rows with missing prices
    df = df.dropna(subset=['price_in_eur'])
    logger.info(f"After removing missing prices: {len(df)} rows")
    
    # Remove price outliers
    if remove_price_outliers:
        df = remove_outliers(df, 'price_in_eur', n_std=outlier_std, use_log=True)
    
    # Apply transformations
    df = create_log_features(df)
    df = create_model_categories(df)
    df = calculate_listing_score(df)
    df = prepare_modeling_features(df)
    
    # Save gold data
    gold_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(gold_path, index=False)
    logger.info(f"Saved {len(df)} rows to gold: {gold_path}")
    
    return df


if __name__ == '__main__':
    from config import LISTINGS_SILVER, LISTINGS_GOLD
    
    logging.basicConfig(level=logging.INFO)
    
    if LISTINGS_SILVER.exists():
        process_silver_to_gold(LISTINGS_SILVER, LISTINGS_GOLD)
    else:
        logger.error(f"Silver file not found: {LISTINGS_SILVER}")
