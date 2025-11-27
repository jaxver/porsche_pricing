"""
Elferspot Listings - Porsche Price Analytics

A professional data science project for analyzing and predicting Porsche listing prices.

Pipeline Stages:
1. Bronze (Raw): data/bronze/
2. Silver (Clean): data/silver/
3. Gold (Model-ready): data/gold/

Modules:
- data_processing: ETL pipeline (Bronze→Silver→Gold)
- models: ML model implementations
- utils: Helper functions (exchange rates, data loading, logging)
"""

__version__ = '1.0.0'
__author__ = 'Jaxver'

from . import data_processing
from . import utils

__all__ = ['data_processing', 'utils']
