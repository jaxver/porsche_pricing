from __future__ import annotations

import pytest

import config


@pytest.fixture(autouse=True)
def _redirect_benchmark_db(tmp_path, monkeypatch):
    monkeypatch.setattr(config, "BENCHMARK_DB", tmp_path / "benchmark_runs.db")
