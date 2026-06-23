from __future__ import annotations

import importlib
import importlib.util

import pandas as pd
import pytest


def test_challengers_module_imports_without_advanced_dependencies():
    module = importlib.import_module("elferspot_listings.modeling.challengers")

    assert hasattr(module, "run_tabpfn_regression")
    assert hasattr(module, "run_autogluon_regression")


def test_run_tabpfn_regression_raises_helpful_error_when_dependency_is_missing():
    if importlib.util.find_spec("tabpfn") is not None:
        pytest.skip("tabpfn is installed in this environment")

    from elferspot_listings.modeling.challengers import run_tabpfn_regression

    X_train = pd.DataFrame({"feature": [1.0, 2.0]})
    y_train = pd.Series([10.0, 20.0])
    X_test = pd.DataFrame({"feature": [3.0]})

    with pytest.raises(RuntimeError, match=r"requirements-advanced\.txt"):
        run_tabpfn_regression(X_train, y_train, X_test)


def test_run_autogluon_regression_raises_helpful_error_when_dependency_is_missing():
    try:
        autogluon_spec = importlib.util.find_spec("autogluon.tabular")
    except ModuleNotFoundError:
        autogluon_spec = None

    if autogluon_spec is not None:
        pytest.skip("autogluon.tabular is installed in this environment")

    from elferspot_listings.modeling.challengers import run_autogluon_regression

    train_df = pd.DataFrame({"feature": [1.0, 2.0], "price_in_eur": [10.0, 20.0]})
    test_df = pd.DataFrame({"feature": [3.0], "price_in_eur": [30.0]})

    with pytest.raises(RuntimeError, match=r"requirements-advanced\.txt"):
        run_autogluon_regression(train_df, test_df, "price_in_eur", ".")
