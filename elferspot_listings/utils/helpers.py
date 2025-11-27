"""
Helper utility functions for data loading, saving, and logging.
"""
import os
import logging
import pandas as pd
from pathlib import Path
from typing import Union, Optional
import sys

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def setup_logging(level: str = 'INFO', log_file: Optional[Path] = None) -> logging.Logger:
    """
    Setup logging configuration.
    
    Args:
        level: Logging level (DEBUG, INFO, WARNING, ERROR, CRITICAL)
        log_file: Optional path to log file
    
    Returns:
        Configured logger
    """
    log_format = '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    handlers = [logging.StreamHandler()]
    
    if log_file:
        log_file.parent.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_file))
    
    logging.basicConfig(
        level=getattr(logging, level.upper()),
        format=log_format,
        handlers=handlers
    )
    
    return logging.getLogger(__name__)


def load_data(
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> pd.DataFrame:
    """
    Load data from various file formats.
    
    Args:
        file_path: Path to data file
        file_type: File type ('csv', 'xlsx', 'parquet'). Auto-detected if None.
        **kwargs: Additional arguments passed to pandas read function
    
    Returns:
        Loaded DataFrame
    """
    file_path = Path(file_path)
    
    if not file_path.exists():
        raise FileNotFoundError(f"File not found: {file_path}")
    
    # Auto-detect file type from extension
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Loading data from {file_path} (type: {file_type})")
    
    if file_type in ['xlsx', 'xls']:
        df = pd.read_excel(file_path, **kwargs)
    elif file_type == 'csv':
        df = pd.read_csv(file_path, **kwargs)
    elif file_type == 'parquet':
        df = pd.read_parquet(file_path, **kwargs)
    elif file_type == 'json':
        df = pd.read_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Loaded {len(df):,} rows and {len(df.columns)} columns")
    return df


def save_data(
    df: pd.DataFrame,
    file_path: Union[str, Path],
    file_type: Optional[str] = None,
    **kwargs
) -> None:
    """
    Save DataFrame to various file formats.
    
    Args:
        df: DataFrame to save
        file_path: Path to save file
        file_type: File type ('csv', 'xlsx', 'parquet'). Auto-detected if None.
        **kwargs: Additional arguments passed to pandas write function
    """
    file_path = Path(file_path)
    file_path.parent.mkdir(parents=True, exist_ok=True)
    
    # Auto-detect file type from extension
    if file_type is None:
        file_type = file_path.suffix.lower().lstrip('.')
    
    logger = logging.getLogger(__name__)
    logger.info(f"Saving data to {file_path} (type: {file_type})")
    
    if file_type in ['xlsx', 'xls']:
        df.to_excel(file_path, index=False, **kwargs)
    elif file_type == 'csv':
        df.to_csv(file_path, index=False, **kwargs)
    elif file_type == 'parquet':
        df.to_parquet(file_path, index=False, **kwargs)
    elif file_type == 'json':
        df.to_json(file_path, **kwargs)
    else:
        raise ValueError(f"Unsupported file type: {file_type}")
    
    logger.info(f"Saved {len(df):,} rows to {file_path}")


def get_project_root() -> Path:
    """Get the project root directory."""
    return PROJECT_ROOT


def ensure_dir(path: Union[str, Path]) -> Path:
    """
    Ensure directory exists, create if it doesn't.
    
    Args:
        path: Directory path
    
    Returns:
        Path object
    """
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path
