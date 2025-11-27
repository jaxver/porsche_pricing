"""
Bronze to Silver data transformation.
Cleans and standardizes raw scraped data.
"""
import pandas as pd
import numpy as np
import logging
import re
from pathlib import Path
from typing import Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from elferspot_listings.utils.exchange_rates import fetch_latest_rates, convert_to_eur

logger = logging.getLogger(__name__)


def clean_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Extract and standardize mileage values."""
    logger.info("Cleaning mileage data")
    
    # Extract mileage value and unit
    df[["mileage_value", "mileage_unit"]] = df["Mileage"].str.extract(
        r'([\d,]+)\s*(km|mi)', expand=True
    )
    
    # Handle fully restored vehicles with no mileage
    mask = (df["mileage_value"].isna()) & (
        df["Condition"].str.contains("fully restored", case=False, na=False)
    )
    df.loc[mask, "mileage_value"] = "1"
    
    # Drop rows with missing mileage
    df = df[~df["mileage_value"].isna()].copy()
    
    # Fill missing units with 'km'
    df["mileage_unit"] = df["mileage_unit"].fillna("km").replace("", "km")
    
    # Convert mileage_value to numeric (remove commas)
    df["mileage_value"] = df["mileage_value"].str.replace(",", "").astype(float)
    
    # Convert miles to kilometers
    df.loc[df["mileage_unit"] == "mi", "mileage_value"] *= 1.60934
    
    # Create standardized Mileage_km column
    df["Mileage_km"] = df["mileage_value"]
    
    logger.info(f"Cleaned mileage for {len(df)} listings")
    return df


def clean_condition(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize condition fields."""
    logger.info("Cleaning condition fields")
    
    df["Condition"] = df["Condition"].apply(
        lambda x: "Unknown" if pd.isna(x) or str(x).strip() == "" else x
    )
    
    df["Paint-to-Sample (PTS)"] = df["Paint-to-Sample (PTS)"].apply(
        lambda x: 1 if str(x).strip().lower() == "yes" else 0
    )
    
    df["Matching numbers"] = df["Matching numbers"].apply(
        lambda x: "Unknown" if pd.isna(x) or str(x).strip() == "" else x
    )
    
    df["Number of vehicle owners"] = df["Number of vehicle owners"].apply(
        lambda x: "Unknown" if pd.isna(x) or str(x).strip() == "" else x
    )
    
    return df


def create_derived_features(df: pd.DataFrame) -> pd.DataFrame:
    """Create derived feature columns."""
    logger.info("Creating derived features")
    
    df['owners_known'] = df['Number of vehicle owners'].apply(
        lambda x: 0 if str(x).strip().lower() == "unknown" else 1
    )
    
    df['is_fully_restored'] = df['Condition'].str.contains(
        "fully restored", case=False, na=False
    ).astype(int)
    
    return df


def standardize_series(df: pd.DataFrame) -> pd.DataFrame:
    """Standardize Series column values."""
    logger.info("Standardizing series values")
    
    # Example: Replace verbose series names with shorter codes
    df['Series'] = df['Series'].replace({
        '982 (718 Boxster/Cayman)': '982',
        '987 (Boxster/Cayman)': '987',
        '981 (Boxster/Cayman)': '981',
    })
    
    return df


def convert_prices_to_eur(df: pd.DataFrame, rates: Optional[dict] = None) -> pd.DataFrame:
    """Convert all prices to EUR."""
    logger.info("Converting prices to EUR")
    
    if rates is None:
        rates = fetch_latest_rates()
    
    # Ensure price and currency columns exist
    if 'price' not in df.columns or 'currency' not in df.columns:
        logger.warning("Price or currency column missing, skipping conversion")
        return df
    
    df['price_in_eur'] = df.apply(
        lambda row: convert_to_eur(row['price'], row['currency'], rates) 
        if pd.notna(row['price']) and pd.notna(row['currency']) 
        else np.nan,
        axis=1
    )
    
    logger.info(f"Converted {df['price_in_eur'].notna().sum()} prices to EUR")
    return df


def process_bronze_to_silver(
    bronze_path: Path,
    silver_path: Path,
    drop_shop_links: bool = True
) -> pd.DataFrame:
    """
    Main function to process Bronze data to Silver.
    
    Args:
        bronze_path: Path to bronze (raw) data file
        silver_path: Path to save silver (cleaned) data
        drop_shop_links: Whether to filter out Elferspot shop links
    
    Returns:
        Cleaned DataFrame
    """
    logger.info(f"Starting Bronze -> Silver transformation")
    logger.info(f"Input: {bronze_path}")
    logger.info(f"Output: {silver_path}")
    
    # Load bronze data
    df = pd.read_excel(bronze_path)
    logger.info(f"Loaded {len(df)} rows from bronze")
    
    # Remove duplicates
    initial_count = len(df)
    df = df.drop_duplicates(subset=['URL'])
    logger.info(f"Removed {initial_count - len(df)} duplicate URLs")
    
    # Drop unnecessary columns
    if 'License documents (Click to open)' in df.columns:
        df = df.drop(columns=['License documents (Click to open)'])
    
    # Filter out shop links if requested
    if drop_shop_links and 'URL' in df.columns:
        initial_count = len(df)
        df = df[~df["URL"].str.contains("https://www.elferspot.com/en/shop/", na=False)]
        logger.info(f"Filtered out {initial_count - len(df)} shop links")
    
    # Apply cleaning functions
    df = clean_mileage(df)
    df = clean_condition(df)
    df = standardize_series(df)
    df = create_derived_features(df)
    df = convert_prices_to_eur(df)
    
    # Save silver data
    silver_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_excel(silver_path, index=False)
    logger.info(f"Saved {len(df)} rows to silver: {silver_path}")
    
    return df


if __name__ == '__main__':
    import sys
    from config import LISTINGS_BRONZE, LISTINGS_SILVER
    
    logging.basicConfig(level=logging.INFO)
    
    if LISTINGS_BRONZE.exists():
        process_bronze_to_silver(LISTINGS_BRONZE, LISTINGS_SILVER)
    else:
        logger.error(f"Bronze file not found: {LISTINGS_BRONZE}")
