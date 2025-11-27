"""
Exchange rate utilities for currency conversion.
"""
import os
import json
import time
import requests
import logging
from pathlib import Path
from typing import Dict, Optional, List

logger = logging.getLogger(__name__)

# Default cache path
DEFAULT_CACHE_PATH = Path(__file__).parent.parent.parent / 'data' / 'exchange_rates.json'
TTL_SECONDS = 24 * 3600  # 24 hours

# Default fallback rates (currency -> EUR multiplier)
STATIC_FALLBACK = {
    'EUR': 1.0,
    'USD': 0.8779,
    'GBP': 1.1870,
    'JPY': 0.006076,
    'CHF': 1.0660,
}


def _load_cache(cache_path: Path) -> Dict[str, float]:
    """Load cached exchange rates if still valid."""
    try:
        if cache_path.exists():
            with open(cache_path, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            ts = obj.get('ts', 0)
            if time.time() - ts < TTL_SECONDS:
                logger.debug(f"Using cached exchange rates from {cache_path}")
                return obj.get('rates', {})
    except Exception as e:
        logger.warning(f"Error loading cache: {e}")
        return {}
    return {}


def _write_cache(rates: Dict[str, float], cache_path: Path) -> None:
    """Write exchange rates to cache file."""
    try:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump({'ts': time.time(), 'rates': rates}, f, indent=2)
        logger.debug(f"Cached exchange rates to {cache_path}")
    except Exception as e:
        logger.warning(f"Error writing cache: {e}")


def fetch_latest_rates(
    base: str = 'EUR',
    symbols: Optional[List[str]] = None,
    use_cache: bool = True,
    cache_path: Optional[Path] = None
) -> Dict[str, float]:
    """
    Fetch latest exchange rates with base 'EUR'.
    
    Returns dict of currency->EUR multiplier.
    If the API fails, returns last cached rates or STATIC_FALLBACK.
    
    Args:
        base: Base currency (default: 'EUR')
        symbols: List of currency symbols to fetch
        use_cache: Whether to use cached rates if available
        cache_path: Path to cache file (default: data/exchange_rates.json)
    
    Returns:
        Dictionary mapping currency codes to EUR conversion rates
    """
    symbols = symbols or ['USD', 'GBP', 'JPY', 'CHF']
    cache_path = cache_path or DEFAULT_CACHE_PATH
    
    # Try cache first
    if use_cache:
        cached = _load_cache(cache_path)
        if cached:
            return cached

    try:
        logger.info(f"Fetching exchange rates from API (base: {base})")
        params = {'base': base, 'symbols': ','.join(symbols)}
        r = requests.get('https://api.exchangerate.host/latest', params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        rates_base = data.get('rates', {})
        
        # Convert rates: data['rates'][C] is units of C per 1 EUR
        # To convert X USD -> EUR: EUR = USD / (USD_per_EUR) = USD * (1 / USD_per_EUR)
        rates_to_eur = {
            c: (1.0 / rates_base[c]) if rates_base.get(c) else STATIC_FALLBACK.get(c, 1.0) 
            for c in rates_base
        }
        rates_to_eur['EUR'] = 1.0
        
        _write_cache(rates_to_eur, cache_path)
        logger.info(f"Successfully fetched {len(rates_to_eur)} exchange rates")
        return rates_to_eur
        
    except Exception as e:
        logger.warning(f"Failed to fetch exchange rates from API: {e}")
        # Fallback to cached or static
        cached = _load_cache(cache_path)
        if cached:
            logger.info("Using cached exchange rates as fallback")
            return cached
        logger.info("Using static fallback exchange rates")
        return STATIC_FALLBACK


def convert_to_eur(amount: float, currency: str, rates: Optional[Dict[str, float]] = None) -> float:
    """
    Convert an amount from given currency to EUR.
    
    Args:
        amount: Amount to convert
        currency: Source currency code (e.g., 'USD', 'GBP')
        rates: Optional exchange rate dictionary. If None, fetches latest rates.
    
    Returns:
        Amount in EUR
    """
    if currency == 'EUR':
        return amount
    
    if rates is None:
        rates = fetch_latest_rates()
    
    rate = rates.get(currency, STATIC_FALLBACK.get(currency, 1.0))
    return amount * rate


if __name__ == '__main__':
    logging.basicConfig(level=logging.INFO)
    rates = fetch_latest_rates()
    print('Currency -> EUR rates:')
    for k, v in rates.items():
        print(f'  {k}: {v:.6f}')
