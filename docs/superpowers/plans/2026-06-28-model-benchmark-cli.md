# Model Benchmark CLI Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add model-scoped benchmark runs plus a CLI entry point for running and reporting single-model or default benchmark selections.

**Architecture:** Keep the existing benchmark runner as the source of truth and add a small model-selection layer that can either preserve the legacy full run or narrow execution to a validated subset. Add a thin argparse CLI that translates command-line flags into runner arguments and prints JSON output for automation.

**Tech Stack:** Python, argparse, pandas, pytest

---

### Task 1: Add model-selection tests

**Files:**
- Modify: `tests/test_train_baselines.py`
- Create: `tests/test_modeling_cli.py`

- [ ] **Step 1: Write the failing runner selection tests**

```python
def test_train_baseline_models_with_ridge_only_runs_ridge(tmp_path, monkeypatch):
    ...

def test_train_baseline_models_with_xgboost_only_runs_xgboost_without_boolean_flag(tmp_path, monkeypatch):
    ...

def test_train_baseline_models_rejects_invalid_model_name(tmp_path):
    ...
```

- [ ] **Step 2: Write the failing CLI test**

```python
def test_cli_parses_arguments_and_prints_json(monkeypatch, capsys, tmp_path):
    ...
```

- [ ] **Step 3: Run the targeted tests**

Run: `rtk proxy "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m pytest tests/test_train_baselines.py tests/test_modeling_cli.py -q`

Expected: fail because the new selection and CLI code does not exist yet.

### Task 2: Implement model selection in `train_baseline_models`

**Files:**
- Modify: `elferspot_listings/modeling/train.py`

- [ ] **Step 1: Add the smallest selection helpers and runner branching**

```python
SUPPORTED_MODELS = {"median", "ridge", "elasticnet", "skrub_ridge", "xgboost", "catboost", "tabpfn", "autogluon", "all"}
DEFAULT_MODELS = ("median", "ridge", "elasticnet", "skrub_ridge")

def train_baseline_models(..., models: list[str] | None = None) -> BenchmarkResult:
    ...
```

- [ ] **Step 2: Run the selection tests again**

Run: `rtk proxy "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m pytest tests/test_train_baselines.py -q`

Expected: the new runner tests pass.

### Task 3: Add the CLI module

**Files:**
- Create: `elferspot_listings/modeling/cli.py`

- [ ] **Step 1: Add argparse entry point and JSON output**

```python
def main(argv: list[str] | None = None) -> int:
    ...

if __name__ == "__main__":
    raise SystemExit(main())
```

- [ ] **Step 2: Run the CLI test**

Run: `rtk proxy "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m pytest tests/test_modeling_cli.py -q`

Expected: pass with mocked I/O.

### Task 4: Final verification and commit

**Files:**
- Modify: any files changed above

- [ ] **Step 1: Run the focused test suite**

Run: `rtk proxy "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m pytest --basetemp "C:\Users\jaxon\AppData\Local\Temp\opencode\pytest" tests\test_train_baselines.py tests\test_modeling_cli.py -q`

- [ ] **Step 2: Commit the atomic change**

```bash
rtk proxy git add elferspot_listings/modeling/train.py elferspot_listings/modeling/cli.py tests/test_train_baselines.py tests/test_modeling_cli.py docs/superpowers/plans/2026-06-28-model-benchmark-cli.md
rtk proxy git commit -m "feat: add model benchmark cli"
```
