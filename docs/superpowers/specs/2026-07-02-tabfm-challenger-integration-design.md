# TabFM Challenger Integration and Run Summary Design

## Summary

Add TabFM as an optional benchmark challenger that runs from the existing CLI benchmark flow and is logged into the SQLite benchmark database like the other models.
Also add a small SQLite query helper that returns the best run per model name by `mae_eur`, while exposing the other metrics and a few run diagnostics for comparison.

## Goals

- Add `tabfm` to the benchmark model set without changing the existing baseline flow.
- Use the official TabFM PyTorch package installation pattern from Google/Hugging Face guidance.
- Keep TabFM behind the repo's `advanced` extra so base installs remain light.
- Log TabFM runs into `benchmark_runs.db` automatically through the existing benchmark DB path.
- Provide a helper that summarizes the best historical run per model type using `mae_eur` as the primary sort key.

## Non-Goals

- No dashboard/UI changes.
- No model training or fine-tuning of TabFM weights. TabFM is used as a zero-shot challenger that fits on the train split and predicts the holdout split.
- No vendoring of the upstream TabFM repository or model weights.

## Dependency And Install Plan

- Update `pyproject.toml` `project.optional-dependencies.advanced` to include the PyTorch TabFM package, using the official extra syntax:
  - `tabfm[pytorch]>=1.0.0`
- Keep the repo-level install instruction as:
  - `python -m pip install -e ".[advanced]"`
- Document that TabFM weights are fetched on first use and that the model package is optional.
- Follow the upstream model guidance in code by loading the PyTorch regression checkpoint through the published TabFM loader, specifically the versioned helper pattern shown in the model card (`from tabfm import TabFMRegressor, tabfm_v1_0_0_pytorch as tabfm_v1_0_0` and `tabfm_v1_0_0.load(model_type="regression")`), not by cloning the upstream repository.
- Keep the existing `advanced` install boundary intact so users who do not need challengers do not pay for TabFM's heavier dependency set.

## Proposed Architecture

### 1. `elferspot_listings/modeling/challengers.py`

- Add `run_tabfm_regression(...)` with the same return shape as the other challenger helpers: `(model, predictions, metadata)`.
- Use the repo's existing feature frame directly, since TabFM can consume mixed numerical and categorical columns from a DataFrame.
- Load the official PyTorch checkpoint for regression from the installed `tabfm` package.
- On missing dependency or model-load failure, raise the repo's existing optional-dependency error wrapper with a helpful install hint.
- Include metadata such as `model_name`, `backend`, `runtime_seconds`, `model_path`, and a note that the first run may download weights.

### 2. `elferspot_listings/modeling/train.py`

- Add `tabfm` to `SUPPORTED_MODEL_NAMES` and ensure it is runnable through `train_baseline_models()` like the existing challengers.
- TabFM should participate in the same train/test split and benchmark DB logging path as the other models.
- The benchmark DB logging should continue to be best-effort and should not fail the whole training run.

### 3. `elferspot_listings/modeling/cli.py`

- Add `tabfm` to `MODEL_CHOICES` so it is selectable from the benchmark CLI.
- No separate backend flag is needed for TabFM in this repo because the integration is PyTorch-only.

### 4. `elferspot_listings/modeling/benchmark_db.py`

- Add a query helper that returns the best run per `model_name`, ranked by:
  - `mae_eur` ascending
  - `created_at` descending as a tiebreaker
  - `run_id` descending as a final tiebreaker
- Include useful run diagnostics alongside the metrics so the summary can explain why a run won.

## Summary Helper Shape

The helper should return one row per model with at least:

- `model_name`
- `run_id`
- `created_at`
- `output_dir`
- `mae_eur`
- `median_ae`
- `mape`
- `within_10`
- `within_15`
- `duration_sec`
- `random_state`
- `train_catboost`
- `run_tabpfn`
- `run_autogluon`
- `autogluon_tl`
- `git_commit`
- a small skipped-model diagnostic, if available, such as skipped count or serialized skipped reasons

This helper is intentionally read-only and can be used by the CLI, dashboard, notebooks, or tests later without changing the benchmark write path.

## Error Handling

- If `tabfm` is missing, skip TabFM with a clear `advanced` install hint.
- If the upstream checkpoint loader or first-use download fails, surface a helpful failure message rather than crashing unrelated benchmark models.
- If the summary helper sees a model with no successful runs, return no row for that model rather than fabricating data.

## Tests

- Add coverage for TabFM challenger wiring and optional-dependency fallback.
- Add CLI coverage that verifies `tabfm` is a valid model choice.
- Add benchmark DB tests for the best-run summary helper, including ordering by `mae_eur` and carrying diagnostic fields.
- Keep tests behavior-focused and use the SQLite DB instead of mocks where practical.

## Acceptance Criteria

- `python -m pip install -e ".[advanced]"` installs the TabFM challenger dependency correctly.
- `tabfm` can be selected from the benchmark CLI and appears in benchmark DB history when it runs.
- The summary helper returns the best historical run per model using `mae_eur` and includes the requested diagnostics.
- Existing benchmark behavior for baselines, TabPFN, AutoGluon, and DB logging remains unchanged.
