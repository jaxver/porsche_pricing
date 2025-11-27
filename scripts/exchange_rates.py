import os
import json
import time
import requests
from datetime import datetime

CACHE_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'exchange_rates.json')
CACHE_PATH = os.path.abspath(CACHE_PATH)
TTL_SECONDS = 24 * 3600  # 24 hours

# Default fallback rates (currency -> EUR multiplier)
STATIC_FALLBACK = {
    'EUR': 1.0,
    'USD': 0.8779,
    'GBP': 1.1870,
    'JPY': 0.006076,
    'CHF': 1.0660,
}


def _load_cache():
    try:
        if os.path.exists(CACHE_PATH):
            with open(CACHE_PATH, 'r', encoding='utf-8') as f:
                obj = json.load(f)
            ts = obj.get('ts', 0)
            if time.time() - ts < TTL_SECONDS:
                return obj.get('rates', {})
    except Exception:
        return {}
    return {}


def _write_cache(rates):
    try:
        os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
        with open(CACHE_PATH, 'w', encoding='utf-8') as f:
            json.dump({'ts': time.time(), 'rates': rates}, f)
    except Exception:
        pass


def fetch_latest_rates(base='EUR', symbols=None, use_cache=True):
    """Fetch latest exchange rates with base 'EUR'. Returns dict currency->EUR multiplier.

    If the API fails, returns last cached rates or STATIC_FALLBACK.
    """
    symbols = symbols or ['USD', 'GBP', 'JPY', 'CHF']
    # Try cache first
    if use_cache:
        cached = _load_cache()
        if cached:
            return cached

    try:
        params = {'base': base, 'symbols': ','.join(symbols)}
        r = requests.get('https://api.exchangerate.host/latest', params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        rates_base = data.get('rates', {})
        # data['rates'][C] is units of C per 1 EUR (if base='EUR'), i.e., 1 EUR = X USD
        # To convert X USD -> EUR: EUR = USD / (USD_per_EUR) = USD * (1 / USD_per_EUR)
        rates_to_eur = {c: (1.0 / rates_base[c]) if rates_base.get(c) else STATIC_FALLBACK.get(c, 1.0) for c in rates_base}
        rates_to_eur['EUR'] = 1.0
        _write_cache(rates_to_eur)
        return rates_to_eur
    except Exception:
        # Fallback to cached or static
        cached = _load_cache()
        if cached:
            return cached
        return STATIC_FALLBACK


if __name__ == '__main__':
    rates = fetch_latest_rates()
    print('Currency -> EUR rates:')
    for k, v in rates.items():
        print(f'{k}: {v}')
