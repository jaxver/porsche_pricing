from elferspot_listings.utils.exchange_rates import convert_to_eur


def test_convert_to_eur_returns_same_amount_for_eur():
    assert convert_to_eur(12345.0, "EUR", rates={"EUR": 1.0}) == 12345.0


def test_convert_to_eur_uses_supplied_rate():
    assert convert_to_eur(100000.0, "USD", rates={"USD": 0.9}) == 90000.0
