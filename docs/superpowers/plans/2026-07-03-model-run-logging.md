# Model Run Logging Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add wrapper-level model run logging so long benchmarks emit heartbeats by default and per-step logs behind `--verbose`.

**Architecture:** Keep logging entirely in `elferspot_listings.modeling.train` and wire a single `--verbose` CLI flag through `elferspot_listings.modeling.cli`. The training wrapper should log model start/end for every model, emit a periodic heartbeat while a model is running, and use extra step logs only when verbose mode is enabled.

**Tech Stack:** Python stdlib `logging`, existing CLI/train wrapper, pytest.

---

### Task 1: Wire `--verbose` through the CLI

**Files:**
- Modify: `elferspot_listings/modeling/cli.py`
- Test: `tests/test_modeling_cli.py`

- [ ] **Step 1: Write the failing test**

```python
def test_cli_passes_verbose_flag_to_train_baseline_models(monkeypatch, capsys, tmp_path):
    from elferspot_listings.modeling import cli

    captured = {}
    monkeypatch.setattr(cli.config, "LISTINGS_GOLD", tmp_path / "default_input.xlsx")
    monkeypatch.setattr(cli.config, "RESULTS_DIR", tmp_path)
    monkeypatch.setattr(cli.pd, "read_excel", lambda path: pd.DataFrame({"price_in_eur": [100000.0], "Mileage_km": [10000.0]}))

    def fake_train_baseline_models(gold_df_arg, output_dir, **kwargs):
        captured["kwargs"] = kwargs
        return SimpleNamespace(metrics={}, output_dir=Path(output_dir), skipped_models={})

    monkeypatch.setattr(cli, "train_baseline_models", fake_train_baseline_models)

    exit_code = cli.main(["--model", "ridge", "--verbose"])

    assert exit_code == 0
    assert captured["kwargs"]["verbose"] is True
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_modeling_cli.py::test_cli_passes_verbose_flag_to_train_baseline_models -q`

- [ ] **Step 3: Write minimal implementation**

```python
parser.add_argument("--verbose", action="store_true", help="Enable detailed model run logging.")

train_kwargs["verbose"] = args.verbose
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_modeling_cli.py::test_cli_passes_verbose_flag_to_train_baseline_models -q`

- [ ] **Step 5: Commit**

```bash
git add elferspot_listings/modeling/cli.py tests/test_modeling_cli.py
git commit -m "feat: add verbose benchmark logging flag"
```

### Task 2: Add wrapper-level heartbeat and step logging

**Files:**
- Modify: `elferspot_listings/modeling/train.py`
- Test: `tests/test_train_baselines.py`

- [ ] **Step 1: Write the failing test**

```python
def test_train_baseline_models_emits_heartbeat_logging(tmp_path, monkeypatch, caplog):
    caplog.set_level(logging.INFO, logger="elferspot_listings.modeling.train")

    def fake_sleep(_seconds):
        return None

    def fake_tabfm(*args, **kwargs):
        return object(), pd.Series([1.0] * len(args[2]), index=args[2].index), {"model_name": "tabfm", "backend": "pytorch", "runtime_seconds": 0.0, "notes": "fake"}

    monkeypatch.setattr("elferspot_listings.modeling.train.run_tabfm_regression", fake_tabfm)
    monkeypatch.setattr("elferspot_listings.modeling.train.time.sleep", fake_sleep)

    train_baseline_models(_gold_frame(), tmp_path, random_state=42, models=["tabfm"], verbose=True)

    assert any("tabfm" in record.message.lower() and "start" in record.message.lower() for record in caplog.records)
    assert any("heartbeat" in record.message.lower() for record in caplog.records)
    assert any("finish" in record.message.lower() for record in caplog.records)
```

- [ ] **Step 2: Run the test to verify it fails**

Run: `pytest tests/test_train_baselines.py::test_train_baseline_models_emits_heartbeat_logging -q`

- [ ] **Step 3: Write minimal implementation**

```python
def _log_model_run(name: str, func: Callable[[], Any], *, verbose: bool, heartbeat_seconds: int = 300) -> Any:
    start = time.perf_counter()
    logger.info("%s: start", name)
    if verbose:
        logger.info("%s: prepare", name)
    next_heartbeat = start + heartbeat_seconds
    result = func()
    while time.perf_counter() >= next_heartbeat:
        logger.info("%s: heartbeat after %.0fs", name, time.perf_counter() - start)
        next_heartbeat += heartbeat_seconds
    logger.info("%s: finish in %.1fs", name, time.perf_counter() - start)
    return result
```

- [ ] **Step 4: Run the test to verify it passes**

Run: `pytest tests/test_train_baselines.py::test_train_baseline_models_emits_heartbeat_logging -q`

- [ ] **Step 5: Commit**

```bash
git add elferspot_listings/modeling/train.py tests/test_train_baselines.py
git commit -m "feat: add benchmark run heartbeats"
```

### Task 3: Verify the end-to-end flow

**Files:**
- Modify: `elferspot_listings/modeling/train.py`
- Modify: `elferspot_listings/modeling/cli.py`
- Test: `tests/test_modeling_cli.py`, `tests/test_train_baselines.py`

- [ ] **Step 1: Run focused tests**

Run:
```bash
pytest tests/test_modeling_cli.py tests/test_train_baselines.py -q
```

- [ ] **Step 2: Confirm logging behavior manually if needed**

Run:
```bash
python -m elferspot_listings.modeling.cli --model tabfm --verbose
```

- [ ] **Step 3: Commit the final integration**

```bash
git add elferspot_listings/modeling/cli.py elferspot_listings/modeling/train.py tests/test_modeling_cli.py tests/test_train_baselines.py
git commit -m "feat: add verbose training logs"
```
