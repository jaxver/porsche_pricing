"""
Silver to Gold data transformation.
Applies feature engineering and prepares data for modeling.
"""
import re
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

TEXT_FEATURE_COLUMNS = ("Title", "Model", "Description", "Secondary_Description")

NON_GENUINE_SUFFIX = r'[\s\-/,:()]*?(?:style|spec|comparison|replica|look|lookalike|inspired|tribute|homage|clone|recreation|restomod|heritage shell)\b'
NON_GENUINE_PREFIX_GUARDS = ''.join(
    f'(?<!\\b{term}\\s)'
    for term in (
        'style',
        'spec',
        'comparison',
        'replica',
        'look',
        'lookalike',
        'inspired',
        'tribute',
        'homage',
        'clone',
        'recreation',
        'restomod',
        'heritage shell',
    )
)
NON_GENUINE_SPECIAL_EDITION_GUARDS = rf'\b(?:speedster|sport classic|dakar)\b(?!{NON_GENUINE_SUFFIX})'
GUARDED_SPECIAL_EDITION_TERMS = rf'{NON_GENUINE_PREFIX_GUARDS}(?:speedster|sport classic|dakar)(?!{NON_GENUINE_SUFFIX})'
GUARDED_GENERIC_RARITY_TERMS = rf'{NON_GENUINE_PREFIX_GUARDS}(?:rare\b|special edition|limited edition)(?!{NON_GENUINE_SUFFIX})'
GUARDED_HERITAGE_DESIGN = rf'{NON_GENUINE_PREFIX_GUARDS}heritage design(?!{NON_GENUINE_SUFFIX})'
NON_DRIVETRAIN_SUFFIX = r'\s+(?:cover|covers?|trim|trims?|noise|rebuild|damage|issues?|problem|problems?|work|service|history|paperwork|records)\b'


def build_listing_text(df: pd.DataFrame) -> pd.Series:
    """Combine available listing text fields into one modeling text source."""
    available_columns = [column for column in TEXT_FEATURE_COLUMNS if column in df.columns]
    if not available_columns:
        return pd.Series("", index=df.index, dtype="object")
    return df[available_columns].fillna("").astype(str).agg(" ".join, axis=1)

MODEL_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(singer|guntherwerks|gunther werks|lanzante|carrera gt)\b", "Bespoke / Rarest Models"),
    (
        r"(gt2 rs|gt2rs|911 gt2 rs|gt2 rsr|911 gt2 rsr|rsr|sport classic|911 st\b|911 s[\s/]?t|60 (jahre|years|anniversary)|911 r\b|le mans centenaire edition|991 club coup[eé]|club coup[eé])",
        "GT2RS and RARE Models",
    ),
    (r"\b(gt3 rs|gt3rs|911 gt3 rs|ruf|dakar|gt2 clubsport)\b", "GT3RS"),
    (
        r"\b(964 carrera rs|993 carrera rs|carrera rs\b|911 carrera rs\b|rs america|911 carrera 2\.7|911 carrera 2,7|911 carrera 2\.7 rs|911 carrera 2\.7 mfi|flachbau|gt4 rs|gt4rs|leichtbau)\b",
        "RS Model",
    ),
    (r"\b(gt3\b(?! rs)|gt2\b(?! rs)|911 gt3\b(?! rs)|911 gt2\b(?! rs)|cup|gt4|911 carrera 3\.2 clubsport)\b", "GT4 / GT3 / GT2"),
    (r"\b(speedster|clubsport|heritage|backdate|restomod|modified|exclusive manufaktur)\b", "Special / Backdate"),
    (r"\b(turbo s|turbo|930)\b", "Turbo S / Turbo"),
    (r"\b(gts)\b", "GTS"),
    (r"\b(911\s+s\b|911\s+sc\b|carrera 3\.0|carrera 3,0|carrera 3\.2|carrera 3,2|911 sc|super carrera|carrera s|carrera 4s)\b", "Carrera 3.0/3.2 / S / SC"),
    (r"\b(boxster|cayman|718|981|982|987)\b", "718"),
    (r"\b(912\b|911\b|911 t\b|911 l\b|911 e\b|911 targa\b|carrera\b|carrera 2\b|cabriolet|targa|coupe|convertible)\b", "Base Carrera / Targa / 912"),
)

MODEL_CATEGORY_ORDER: tuple[str, ...] = (
    "Base Carrera / Targa / 912",
    "Carrera 3.0/3.2 / S / SC",
    "GTS",
    "Turbo S / Turbo",
    "GT4 / GT3 / GT2",
    "Special / Backdate",
    "RS Model",
    "GT3RS",
    "GT2RS and RARE Models",
    "Bespoke / Rarest Models",
    "718",
    "Other",
)


def remove_outliers(
    df: pd.DataFrame,
    column: str,
    n_std: float = 3.0,
    use_log: bool = False
) -> pd.DataFrame:
    """Remove outliers based on standard deviations."""
    logger.info(f"Removing outliers from {column} (±{n_std} std)")

    initial_count = len(df)
    if initial_count < 2:
        return df.copy()
    
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
    percent_removed = 0.0 if initial_count == 0 else removed / initial_count * 100
    logger.info(f"Removed {removed} outliers ({percent_removed:.1f}%)")
    
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
    """Create legacy model categories for better price generalization."""
    logger.info("Creating model categories")

    result = df.copy()

    def categorize_model(model: object) -> str:
        if pd.isna(model):
            return "Other"

        model_text = str(model)
        for pattern, category in MODEL_CATEGORY_RULES:
            if re.search(pattern, model_text, flags=re.IGNORECASE):
                return category
        return "Other"

    if "Model" not in result.columns:
        logger.warning("Model column not found, assigning Other to model categories")
        result["model_category"] = "Other"
        return result

    result["model_category"] = result["Model"].apply(categorize_model)

    logger.info("Created %s model categories", result["model_category"].nunique())
    return result


def add_legacy_model_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add ordered model category and mileage interaction features."""
    logger.info("Adding legacy model interaction features")

    result = df.copy()
    if "model_category" not in result.columns:
        result = create_model_categories(result)

    categories = pd.Categorical(
        result["model_category"],
        categories=MODEL_CATEGORY_ORDER,
        ordered=True,
    )
    ordered_codes = pd.Series(categories.codes, index=result.index)
    ordered_codes = ordered_codes.where(ordered_codes != -1, len(MODEL_CATEGORY_ORDER) - 1)
    result["model_cat_ordered"] = ordered_codes.astype(int)

    mileage = (
        pd.to_numeric(result["Mileage_km"], errors="coerce")
        if "Mileage_km" in result.columns
        else pd.Series(np.nan, index=result.index, dtype="float64")
    )
    if "Mileage_sq" in result.columns:
        mileage_sq = pd.to_numeric(result["Mileage_sq"], errors="coerce")
    else:
        mileage_sq = mileage**2

    result["inv_mileage"] = 1 / (mileage + 1)
    result["Mileage_model_cat"] = mileage * result["model_cat_ordered"]
    result["inv_Mileage_model_cat"] = result["inv_mileage"] * result["model_cat_ordered"]
    result["Mileage_sq_model_cat"] = mileage_sq * result["model_cat_ordered"]
    return result


def add_legacy_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy normalized binary features used by historical modeling notebooks."""
    result = df.copy()

    if "Ready to drive" in result.columns:
        ready = result["Ready to drive"]
    else:
        ready = pd.Series("", index=result.index)

    if "Drive" in result.columns:
        drive = result["Drive"]
    else:
        drive = pd.Series("", index=result.index)

    if "Matching numbers" in result.columns:
        matching = result["Matching numbers"]
    else:
        matching = pd.Series("", index=result.index)

    ready = ready.fillna("").astype(str).str.strip().str.lower()
    drive = drive.fillna("").astype(str).str.strip().str.lower()
    matching = matching.fillna("").astype(str).str.strip().str.lower()

    result["state_yes"] = ready.isin({"yes", "y", "true", "1"}).astype(int)
    result["state_Rear drive"] = drive.str.contains(r"\b(?:rear|rwd)\b", regex=True).astype(int)
    result["matching_yes"] = matching.str.contains(r"\b(?:yes|matching numbers|numbers matching|matching drivetrain)\b", regex=True).astype(int)
    return result


def calculate_listing_score(df: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate a quality score for each listing based on completeness
    and description-derived features.
    Higher score = more complete/quality listing data.
    """
    logger.info("Calculating listing quality scores")
    
    score_components = []
    
    # --- Data completeness scoring ---
    if 'Matching numbers' in df.columns:
        matching_numbers = df['Matching numbers'].fillna('').astype(str).str.strip().str.lower()
        score_components.append(
            matching_numbers.isin({'yes', 'matching numbers', 'numbers matching', 'matching drivetrain'}).astype(int) * 10
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
    
    if 'Mileage_km' in df.columns:
        mileage_score = np.where(df['Mileage_km'] < 50000, 10, 0)
        score_components.append(pd.Series(mileage_score, index=df.index))
    
    # --- Description-derived feature extraction ---
    description_patterns = {
        'restoration_full': r'\b(?:frame[- ]?off|body[- ]?off|complete restoration|fully restored|fully rebuilt|nut and bolt (?:restoration|rebuild)|completely restored|full restoration|ground[- ]?up restoration|restored to (?:original|factory) specification|mechanically restored|interior refurbishment|engine overhaul)\b',
        'restoration_partial': r'\b(?:partial restoration|cosmetic refresh|lightly restored|restored in parts)\b',
        'is_restomod': r'\b(?:backdate|restomod|modified|custom (?:build|interior|paint|exhaust|engine|body))\b',
        'has_docs': r'\b(?:full (?:documentation|service history|records|history)|extensive records|well documented|fully documented)\b',
        'is_matching_numbers': r'\b(?:matching numbers|numbers matching|matching drivetrain)\b',
        'is_mint': r'\b(?:mint condition|collector quality|fully sorted|excellent condition|top condition|showroom condition)\b',
        'is_race_ready': r'\b(?:rally ready|race[- ]?ready|track[- ]?prepped|bucket seats|racing harness|homologated|fire suppression system|racing history|race winner|overall winner|group winner|24 hour|nürburgring|nurburgring|spa|le mans|adac|fia|championship|podium|cup\s*s?\b|clubsport|club sport|cup car)\b',
         'is_rare': rf'\b(?:{GUARDED_GENERIC_RARITY_TERMS}|1 of (?:\d+|one)|one of \d+|only \d+ (?:produced|built|made|delivered|examples|cars)|\d+ produced|production number \d+|number \d+ of \d+|rwb|singer|guntherwerks|elfwerks|{GUARDED_SPECIAL_EDITION_TERMS}|911 r\b|{NON_GENUINE_PREFIX_GUARDS}911 st\b(?!{NON_GENUINE_SUFFIX})|{NON_GENUINE_PREFIX_GUARDS}s/t\b(?!{NON_GENUINE_SUFFIX})|{GUARDED_HERITAGE_DESIGN}|bespoke|factory-approved exclusivity|exclusive manufaktur|paint to sample|pts|sonderwunsch|unique piece|unique example|special commission|commissioned)\b',
        'rsr_st_special': rf'{NON_GENUINE_PREFIX_GUARDS}\b(?:porsche\s+911\s+rsr[\s\-/]+st\b|911\s+rsr[\s\-/]+st\b|rsr[\s\-/]+st\b)(?!{NON_GENUINE_SUFFIX})',
        'is_accident_free': r'\b(?:accident[- ]?free|never crashed|clean title|no accidents|undamaged)\b',
        'has_upgrades': r'\b(?:KW suspension|x51|upgraded brakes|recaro|limited[- ]?slip|aftermarket (?:exhaust|turbo|suspension|intake|wheels)|performance parts|turbo upgrade|weissach package)\b',
        'first_owner': r'\b(?:first owner|one owner|single owner|original owner|first hand|single registered keeper)\b',
        'limited_production': r'\b(?:one of|1 of|only)\s+\d+\s+(?:produced|built|made|delivered|examples|cars)\b|\b\d+\s+produced\b|\bnumber\s+\d+\s+of\s+\d+\b',
        'racing_history': r'\b(?:racing history|race winner|overall winner|group winner|24 hour|nürburgring|nurburgring|spa|le mans|adac|fia|championship|podium)\b',
        'specialist_build': r'\b(?:rs tuning|manthey|ruf|singer|gunther|guntherwerks|lanzante|kremer|dp motorsport|techart|gemballa|rwb|theon|paul stephens|tuthill|emerson|canepa)\b',
        'bespoke_exclusive': r'\b(?:bespoke|factory-approved exclusivity|exclusive manufaktur|paint to sample|pts|sonderwunsch|unique piece|unique example|special commission|commissioned|commission)\b',
        'zero_running_hours': r'\b(?:zero running hours|0 running hours|no running hours)\b',
        'engine_transmission_rebuilt': r'\b(?:engine and transmission|gearbox).{0,80}\b(?:overhauled|rebuilt|rebuild|restored)\b|\b(?:overhauled|rebuilt|rebuild|restored).{0,80}\b(?:engine and transmission|gearbox)\b',
        'non_rebuilt': r'\bunrestored\b|\b(?:not|never|non[- ]?)\s*(?:rebuilt|restored|overhauled)\b',
        'needs_rebuild': r'\b(?:needs?|requires?|requiring|awaiting|due for)(?!\s+no\b)\s+(?:an?\s+)?(?:(?:engine|gearbox|transmission|mechanical)\s+)?(?:rebuild|overhaul|restoration|recommissioning|work)\b',
        'body_only': r'\b(?:body[- ]?only|shell[- ]?only|bare shell|rolling shell|rolling chassis|rolling project|roller project|chassis only)\b',
        'missing_drivetrain': rf'\bmissing\s+(?:the\s+)?(?:engine|gearbox|transmission|drivetrain|motor)\b(?!{NON_DRIVETRAIN_SUFFIX})|\b(?:engine|gearbox|transmission|drivetrain)\s+(?:missing|absent|not included)\b(?!{NON_DRIVETRAIN_SUFFIX})|\bwithout\s+(?:the\s+)?(?:engine|gearbox|transmission|drivetrain|motor)\b(?!{NON_DRIVETRAIN_SUFFIX})',
        'project_car': r'\b(?:project car|restoration project|unfinished project|parts car|for restoration|requires restoration|requires recommissioning|needs recommissioning)\b',
         'not_ready_to_drive_text': r'\b(?:not ready to drive|not[\s-]+running|not[\s-]+drivable|non[- ]running|does not run|doesn\'?t run|will not start|won\'?t start|does(?:\s+not|n\'?t)\s+start|cannot start|not roadworthy|no\s+(?:mot|tuv|tüv)\b|without\s+(?:mot|tuv|tüv)\b|no\s+(?:mot|tuv|tüv)\s+certificate|without\s+(?:mot|tuv|tüv)\s+certificate)\b',
         'accident_damage': r'(?<!no\s)(?<!without\s)(?<!no known\s)(?<!without known\s)\b(?:accident damage|damaged car|salvage title|write[- ]off|crash damage|fire damage|flood damage)\b',
        'cup_clubsport': r'\b(?:cup\s*s?\b|clubsport|club sport|cup car)\b',
         'heritage_special': rf'\b(?:{GUARDED_SPECIAL_EDITION_TERMS}|911 r\b|{NON_GENUINE_PREFIX_GUARDS}911 st\b(?!{NON_GENUINE_SUFFIX})|{NON_GENUINE_PREFIX_GUARDS}s/t\b(?!{NON_GENUINE_SUFFIX})|{GUARDED_HERITAGE_DESIGN})\b',
        'weissach_package': r'\bweissach package\b',
        'pccb': r'\bpccb\b',
        'ceramic_brakes': r'\b(?:ceramic brakes?|ceramic disc brakes?)\b',
        'bucket_seats': r'\bbucket seats?\b',
        'clubsport_package': r'\bclubsport package\b|\bclub sport package\b',
        'front_axle_lift': r'\b(?:front axle lift(?: system)?|front lift system)\b',
        'sport_chrono': r'\b(?:sport chrono(?: package)?|chrono package)\b',
        'manual_transmission_text': r'\b(?:\d[- ]?speed manual|manual transmission|manual gearbox)\b',
        'paint_to_sample_text': r'\bpaint[- ]to[- ]sample\b|\bpts\b|\bsonderwunsch\b',
        'manthey': r'\bmanthey\b',
        'ruf': r'\bruf\b',
        'techart': r'\btechart\b',
        'carbon_package': r'\bcarbon package\b',
        'lightweight_package': r'\blightweight package\b',
        'full_leather': r'\b(?:full leather(?: interior)?|extended leather package)\b',
        'carbon_bucket_seats': r'\b(?:carbon(?:[ -](?:fiber|fibre))?\s+bucket seats?|carbon buckets?)\b',
    }
    
    description_weights = {
        'restoration_full': 2.0,
        'restoration_partial': 1.5,
        'is_restomod': 1.7,
        'has_docs': 0.7,
        'is_matching_numbers': 1.0,
        'is_mint': 0.5,
        'is_race_ready': 2.5,
        'is_rare': 2.5,
        'rsr_st_special': 2.0,
        'is_accident_free': 0.5,
        'has_upgrades': 2.3,
        'first_owner': 1.2,
        'limited_production': 2.0,
        'racing_history': 2.0,
        'specialist_build': 1.8,
        'bespoke_exclusive': 1.8,
        'zero_running_hours': 1.8,
        'engine_transmission_rebuilt': 1.2,
        'cup_clubsport': 1.5,
        'heritage_special': 2.0,
        'weissach_package': 1.8,
        'pccb': 1.8,
        'ceramic_brakes': 1.8,
        'bucket_seats': 1.4,
        'clubsport_package': 1.8,
        'front_axle_lift': 1.5,
        'sport_chrono': 1.4,
        'manual_transmission_text': 1.4,
        'paint_to_sample_text': 1.8,
        'manthey': 1.7,
        'ruf': 1.7,
        'techart': 1.7,
        'carbon_package': 1.5,
        'lightweight_package': 1.5,
        'full_leather': 1.0,
        'carbon_bucket_seats': 1.8,
    }

    negative_condition_weights = {
        'non_rebuilt': -1.2,
        'needs_rebuild': -2.5,
        'body_only': -4.0,
        'missing_drivetrain': -4.0,
        'project_car': -2.5,
        'not_ready_to_drive_text': -2.0,
        'accident_damage': -2.5,
    }

    if any(column in df.columns for column in TEXT_FEATURE_COLUMNS):
        df['listing_text'] = build_listing_text(df)
        text_series = df['listing_text'].fillna("").str.lower()
        for feature, pattern in description_patterns.items():
            df[feature] = text_series.str.contains(pattern, regex=True, flags=re.IGNORECASE).astype(int)
            if feature in negative_condition_weights:
                continue
            score_components.append(df[feature] * description_weights[feature])

        negative_penalty = pd.Series(0.0, index=df.index)
        for feature, weight in negative_condition_weights.items():
            negative_penalty = negative_penalty + (df[feature] * weight)
        score_components.append(negative_penalty.clip(lower=-12.0))
    
    # --- Combine scores ---
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


def add_price_inflation_feature(df: pd.DataFrame, annual_rate: float = 0.02) -> pd.DataFrame:
    """Add an inflation multiplier based on how old the scrape is within the dataset."""
    df = df.copy()
    if "Scraped_At" not in df.columns:
        df["price_inflation_factor"] = 1.0
        return df

    scrape_dates = pd.to_datetime(df["Scraped_At"], errors="coerce")
    latest_scrape = scrape_dates.max()
    if pd.isna(latest_scrape):
        df["price_inflation_factor"] = 1.0
        return df

    days_since_scrape = (latest_scrape - scrape_dates).dt.total_seconds() / 86400.0
    df["price_inflation_factor"] = (1.0 + annual_rate) ** (days_since_scrape / 365.25)
    df["price_inflation_factor"] = df["price_inflation_factor"].fillna(1.0)
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
    df = add_price_inflation_feature(df)
    df = create_model_categories(df)
    df = add_legacy_model_interaction_features(df)
    df = calculate_listing_score(df)
    df = add_legacy_binary_flags(df)
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
