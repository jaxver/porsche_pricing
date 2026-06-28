from __future__ import annotations

import tomllib
from pathlib import Path


def test_pyproject_defines_core_and_optional_dependency_groups():
    pyproject = tomllib.loads(Path("pyproject.toml").read_text(encoding="utf-8"))

    project = pyproject["project"]
    dependencies = set(project["dependencies"])
    optional_dependencies = project["optional-dependencies"]

    assert "pandas>=1.5.0" in dependencies
    assert "scikit-learn>=1.2.0" in dependencies
    assert "pytest>=7.2.0" not in dependencies

    assert "tabpfn>=2.0.0" in optional_dependencies["advanced"]
    assert "tabpfn-client>=0.1.0" in optional_dependencies["advanced"]
    assert "autogluon.tabular>=1.2.0" in optional_dependencies["advanced"]
    assert "xgboost>=2.0.0" in optional_dependencies["advanced"]
    assert "perpetual>=0.4.0" in optional_dependencies["advanced"]

    assert "pytest>=7.2.0" in optional_dependencies["dev"]
    assert "pytest>=7.2.0" in optional_dependencies["ci"]
