# TabFM Challenger Integration Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add TabFM as an optional benchmark challenger, expose it through the CLI, log it into the SQLite benchmark history, and provide a summary helper that returns the best historical run per model by `mae_eur`.

**Architecture:** Keep the existing benchmark split/test flow and add TabFM as another challenger inside `elferspot_listings/modeling/challengers.py`. Extend the benchmark run schema so the SQLite history knows whether TabFM ran, then add a read-only summary query over `benchmark_db.py` that surfaces the best run per model with diagnostics. Keep the dependency optional via the existing `advanced` extra so the default install stays light.

**Tech Stack:** Python 3.13, `sqlite3`, `pandas`, `scikit-learn`, upstream `tabfm[pytorch]` optional dependency, `pytest`.

---

### Task 1: Add the TabFM install dependency and document it

**Files:**
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `tests/test_pyproject.py`

- [ ] **Step 1: Write the failing test**

Update `tests/test_pyproject.py` so the advanced extra must include the TabFM PyTorch package:

```python
assert "tabfm[pytorch]>=1.0.0" in optional_dependencies["advanced"]
```

Also update the README expectation so the advanced benchmark section names TabFM alongside TabPFN and AutoGluon.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_pyproject.py -v`
Expected: fail because `tabfm[pytorch]>=1.0.0` is not yet in `pyproject.toml`.

- [ ] **Step 3: Write minimal implementation**

Update `pyproject.toml`:

```toml
[project.optional-dependencies]
advanced = [
    "tabfm[pytorch]>=1.0.0",
    "tabpfn>=2.0.0",
    "tabpfn-client>=0.1.0",
    "autogluon.tabular>=1.2.0",
    "xgboost>=2.0.0",
    "perpetual>=0.4.0",
    "lightgbm>=4.0.0",
    "tabicl>=2.0.0",
    "torch>=2.5.0",
    "torchvision>=0.20.0",
    "torchaudio>=2.5.0",
]
```

Update `README.md` so the optional benchmark section says TabFM is included in the advanced challenger install path and that first use downloads the published weights.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_pyproject.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add -f pyproject.toml README.md tests/test_pyproject.py
rtk proxy git commit -m "feat: add tabfm advanced dependency"
```

---

### Task 2: Add the TabFM challenger and wire it into training and CLI selection

**Files:**
- Modify: `elferspot_listings/modeling/challengers.py`
- Modify: `elferspot_listings/modeling/train.py`
- Modify: `elferspot_listings/modeling/cli.py`
- Modify: `elferspot_listings/modeling/benchmark_db.py`
- Modify: `tests/test_challengers.py`
- Modify: `tests/test_train_baselines.py`
- Modify: `tests/test_modeling_cli.py`
- Modify: `tests/test_benchmark_db.py`

- [ ] **Step 1: Write the failing tests**

Add a success-path challenger test that stubs the public TabFM API exactly as the model card describes:

```python
def test_run_tabfm_regression_returns_predictions_and_metadata(monkeypatch):
    from elferspot_listings.modeling.challengers import run_tabfm_regression

    class FakeVersionedCheckpoint:
        def load(self, model_type="regression"):
            assert model_type == "regression"
            return object()

    class FakeTabFMRegressor:
        def __init__(self, model):
            self.model = model

        def fit(self, X_train, y_train):
            self.fit_shape = (len(X_train), len(y_train))

        def predict(self, X_test):
            return pd.Series([123.0] * len(X_test), index=X_test.index)

    fake_tabfm = types.ModuleType("tabfm")
    fake_tabfm.TabFMRegressor = FakeTabFMRegressor
    fake_tabfm.tabfm_v1_0_0_pytorch = FakeVersionedCheckpoint()
    monkeypatch.setitem(sys.modules, "tabfm", fake_tabfm)

    model, predictions, metadata = run_tabfm_regression(X_train, y_train, X_test)

    assert isinstance(model, FakeTabFMRegressor)
    assert list(predictions["predicted_price_eur"]) == [123.0]
    assert metadata["model_name"] == "tabfm"
    assert metadata["backend"] == "pytorch"
```

Add a training test that proves `tabfm` does not run by default and can run when explicitly requested:

```python
monkeypatch.setattr("elferspot_listings.modeling.train.run_tabfm_regression", lambda *_args, **_kwargs: (_ for _ in ()).throw(AssertionError("tabfm should not run")))
```

Then add a `models=["tabfm"]` case that returns fake predictions and verifies the run is logged to SQLite with a `tabfm` metric row.

Add CLI coverage for the model selector:

```python
exit_code = cli.main(["--model", "tabfm"])
assert captured["kwargs"]["models"] == ["tabfm"]
```

And update the existing `--model all --include-optionals` test so it also expects `run_tabfm` to be forwarded as `True`.

- [ ] **Step 2: Run the tests to verify they fail**

Run:

```powershell
pytest tests/test_challengers.py -v
pytest tests/test_train_baselines.py -v
pytest tests/test_modeling_cli.py -v
```

Expected: TabFM-related tests fail because the helper, training branch, and CLI choice do not exist yet.

- [ ] **Step 3: Write minimal implementation**

In `elferspot_listings/modeling/challengers.py`, add a TabFM helper that follows the published loader pattern and uses the repo's existing DataFrame feature frame directly:

```python
from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0

checkpoint = tabfm_v1_0_0.load(model_type="regression")
model = TabFMRegressor(model=checkpoint)
model.fit(X_train, y_train)
predictions = model.predict(X_test)
```

Return metadata that includes `model_name="tabfm"`, `backend="pytorch"`, `runtime_seconds`, and a short note that the first run may download weights.

In `elferspot_listings/modeling/train.py`:

```python
SUPPORTED_MODEL_NAMES = {
    "median",
    "ridge",
    "elasticnet",
    "skrub_ridge",
    "xgboost",
    "perpetual",
    "catboost",
    "tabpfn",
    "tabfm",
    "autogluon",
    "all",
}

def train_baseline_models(
    gold_df: pd.DataFrame,
    output_dir: str | Path,
    random_state: int = 42,
    train_catboost: bool = False,
    tune_elasticnet: bool = False,
    tune_catboost: bool = False,
    tuning_trials: int = 25,
    run_xgboost: bool = False,
    run_perpetual: bool = False,
    run_tabpfn: bool = False,
    run_tabfm: bool = False,
    tabpfn_model_paths: list[str | None] | None = None,
    tabpfn_backend: str = "local",
    tabpfn_thinking: bool = False,
    tabpfn_thinking_effort: str = "medium",
    tabpfn_thinking_timeout: float | int | None = None,
    tabpfn_thinking_metric: str = "rmse",
    run_autogluon: bool = False,
    autogluon_time_limit: int = 600,
    autogluon_presets: str = "best_quality",
    autogluon_dynamic_stacking: bool | None = None,
    autogluon_clean_output: bool = False,
    models: list[str] | None = None,
    device: str = "cpu",
    gpu_devices: str | None = None,
):
    should_run_tabfm = _should_run_model(requested_models, "tabfm", legacy_enabled=run_tabfm)
```

Add the TabFM branch alongside the other challengers, append its predictions to the benchmark output, and include a `run_tabfm` boolean in the SQLite logging payload.

Update the SQLite schema and insert path so the `runs` row captures whether TabFM ran:

```python
CREATE TABLE IF NOT EXISTS runs (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    created_at      TEXT NOT NULL DEFAULT (datetime('now')),
    random_state    INTEGER NOT NULL DEFAULT 42,
    train_catboost  INTEGER NOT NULL DEFAULT 0,
    run_tabpfn      INTEGER NOT NULL DEFAULT 0,
    run_tabfm       INTEGER NOT NULL DEFAULT 0,
    run_autogluon   INTEGER NOT NULL DEFAULT 0,
    autogluon_tl    INTEGER NOT NULL DEFAULT 600,
    output_dir      TEXT,
    duration_sec    REAL,
    git_commit      TEXT
);

benchmark_db.insert_run(
    config.BENCHMARK_DB,
    random_state=random_state,
    train_catboost=train_catboost,
    run_tabpfn=tabpfn_ran,
    run_tabfm=tabfm_ran,
    run_autogluon=should_run_autogluon,
    autogluon_tl=autogluon_time_limit,
    output_dir=output_path,
    duration_sec=time.perf_counter() - start_time,
)
```

Update `get_latest_run()` and `get_run_history()` so their returned rows include `run_tabfm` as part of the benchmark diagnostics.

In `elferspot_listings/modeling/cli.py`, add `tabfm` to `MODEL_CHOICES` and forward `run_tabfm: include_optionals` in `train_kwargs`.

- [ ] **Step 4: Run the tests to verify they pass**

Run:

```powershell
pytest tests/test_challengers.py -v
pytest tests/test_train_baselines.py -v
pytest tests/test_modeling_cli.py -v
```

Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add -f elferspot_listings/modeling/challengers.py elferspot_listings/modeling/train.py elferspot_listings/modeling/cli.py tests/test_challengers.py tests/test_train_baselines.py tests/test_modeling_cli.py
rtk proxy git commit -m "feat: add tabfm challenger support"
```

---

### Task 3: Add the best-run summary helper to the benchmark database

**Files:**
- Modify: `elferspot_listings/modeling/benchmark_db.py`
- Modify: `tests/test_benchmark_db.py`

- [ ] **Step 1: Write the failing test**

Add a test that creates multiple runs for the same `model_name` and verifies the helper picks the row with the lowest `mae_eur`, then the newest `created_at`, then the newest `run_id`.

The helper should return a `pandas.DataFrame` with at least these columns:

```python
[
    "model_name",
    "run_id",
    "created_at",
    "output_dir",
    "mae_eur",
    "median_ae",
    "mape",
    "within_10",
    "within_15",
    "duration_sec",
    "random_state",
    "train_catboost",
    "run_tabpfn",
    "run_tabfm",
    "run_autogluon",
    "autogluon_tl",
    "git_commit",
    "skipped_count",
    "skipped_models",
]
```

Use real SQLite writes in the test and assert the helper returns one row per model.

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_benchmark_db.py -v`
Expected: fail because the summary helper does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Add a read-only summary helper in `benchmark_db.py` that uses a window function to rank metrics by model:

```sql
ROW_NUMBER() OVER (
    PARTITION BY m.model_name
    ORDER BY m.mae_eur ASC, r.created_at DESC, r.id DESC
) AS rank
```

Join in skipped-model diagnostics with `COUNT(*) AS skipped_count` and `GROUP_CONCAT(model_name, ', ') AS skipped_models`.

Return the best row per model as a `pandas.DataFrame`, ordered by `model_name`.

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_benchmark_db.py -v`
Expected: PASS.

- [ ] **Step 5: Commit**

```bash
rtk proxy git add -f elferspot_listings/modeling/benchmark_db.py tests/test_benchmark_db.py
rtk proxy git commit -m "feat: add benchmark best-run summary"
```

---

### Task 4: Verify the end-to-end benchmark path and clean up any regressions

**Files:**
- Modify: `tests/test_train_baselines.py`
- Modify: `tests/test_modeling_cli.py`

- [ ] **Step 1: Add a cross-check test for SQLite history shape**

After a `models=["tabfm"]` training run, assert that `benchmark_db.get_latest_run(db_path)` includes:

```python
assert latest["run_tabfm"] == 1
assert latest["metrics"]["tabfm"]["mae_eur"] == 123.0
```

- [ ] **Step 2: Run the focused benchmark tests**

Run:

```powershell
pytest tests/test_train_baselines.py -v
pytest tests/test_modeling_cli.py -v
pytest tests/test_benchmark_db.py -v
```

Expected: PASS.

- [ ] **Step 3: Run the full local test slice for the touched area**

Run:

```powershell
pytest tests/test_challengers.py tests/test_train_baselines.py tests/test_modeling_cli.py tests/test_benchmark_db.py tests/test_pyproject.py -v
```

Expected: PASS.

- [ ] **Step 4: Commit**

```bash
rtk proxy git add -f tests/test_train_baselines.py tests/test_modeling_cli.py
rtk proxy git commit -m "test: cover tabfm benchmark wiring"
```
