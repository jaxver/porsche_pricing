# Legacy Feature And Modeling Parity Design

## Goal

Restore the predictive feature engineering and reusable modeling behavior that existed in legacy notebooks but is missing or unused in the production Python pipeline.

The immediate motivation is the benchmark failure mode where rare and high-value Porsche listings are systematically underpriced. The legacy notebooks had stronger model hierarchy features, ordinal encodings, and mileage/category interactions that likely addressed this better than the current simplified production mapping.

## Scope

This work covers production data/modeling parity only:

- Restore legacy feature engineering in the Bronze/Silver/Gold production modules where appropriate.
- Make restored features available to the benchmark/modeling feature selector.
- Add reusable CatBoost quantile interval modeling support from the legacy quantile notebook.
- Add tests that lock in representative legacy behavior.

This work does not productionize market-insight dashboards, VIF notebooks, or exploratory plotting/reporting unless directly needed by model training.

## Current Gap

The current `silver_to_gold.py` maps `Model` to only `911`, `718`, or `Other`. The legacy notebook used a richer hierarchy:

- `Bespoke / Rarest Models`
- `GT2RS and RARE Models`
- `GT3RS`
- `RS Model`
- `GT4 / GT3 / GT2`
- `Special / Backdate`
- `Turbo S / Turbo`
- `GTS`
- `Carrera 3.0/3.2 / S / SC`
- `Base Carrera / Targa / 912`

The legacy notebook also generated:

- `model_cat_ordered`
- `inv_mileage`
- `Mileage_model_cat`
- `inv_Mileage_model_cat`
- `Mileage_sq_model_cat`

Some description-derived flags still exist, but they are only folded into `listing_score`; they are not selected directly as model inputs.

## Data Feature Design

### Model hierarchy

Move the legacy category matching into production code as a deterministic helper. Matching should be ordered from most specific/high-value categories to broad categories so terms like `GT2 RS`, `Carrera RS`, `Speedster`, `Singer`, and `Turbo S` do not collapse into generic `911` or `Other`.

### Ordinal and interaction features

Gold output should include:

- `model_cat_ordered`: ordered category code based on the legacy hierarchy.
- `inv_mileage`: `1 / (Mileage_km + 1)`.
- `Mileage_model_cat`: `Mileage_km * model_cat_ordered`.
- `inv_Mileage_model_cat`: `inv_mileage * model_cat_ordered`.
- `Mileage_sq_model_cat`: `Mileage_sq * model_cat_ordered`.

These features must tolerate missing mileage and unknown categories without crashing.

### Description flags

The existing description-derived binary flags should stay in Gold and be directly available as model features, not only as components of `listing_score`.

Priority flags include:

- `is_rare`
- `is_restomod`
- `is_race_ready`
- `restoration_full`
- `restoration_partial`
- `has_docs`
- `is_matching_numbers`
- `is_mint`
- `is_accident_free`
- `has_upgrades`
- `first_owner`

### Legacy normalized categoricals

Add or preserve production equivalents for legacy modeling columns where useful:

- `state_yes`: normalized ready-to-drive indicator.
- `state_Rear drive`: normalized rear-drive indicator.
- `matching_yes`: normalized matching-numbers indicator.

These should be numeric binary features so linear models and tree models can use them consistently.

## Modeling Feature Selection

Extend `elferspot_listings/modeling/features.py` so the restored numeric and binary features are selected when present.

Keep the existing categorical features, including `Model`, `Series`, and `model_category`.

Do not remove current production features unless a test or benchmark shows a concrete reason.

## CatBoost Quantile Modeling

Add a reusable production helper for legacy CatBoost interval modeling:

- Train three CatBoost regressors on log price with quantile losses: `alpha=0.05`, `alpha=0.5`, and `alpha=0.95`.
- Return EUR-space predictions as `pred_lower`, `pred_price`, and `pred_upper`.
- Use the same selected feature frame and categorical-feature indexing as the existing CatBoost path.
- Support CPU/GPU params consistently with current CatBoost helpers.

This can be exposed as a helper first. CLI exposure can be a separate commit if needed.

## Testing Strategy

Add focused tests before implementation changes:

- Model hierarchy tests for representative examples: `GT2 RS`, `Carrera RS`, `GT3 RS`, `Speedster`, `Singer`, `Turbo S`, `GTS`, regular Carrera, Boxster/Cayman fallback.
- Feature formula tests for `model_cat_ordered`, `inv_mileage`, and mileage/category interactions.
- Feature selector tests proving restored columns are included when present.
- CatBoost quantile smoke test that skips cleanly if CatBoost is unavailable.

Regression tests should use small synthetic frames and avoid production data files.

## Commit Plan

Use atomic semantic commits:

- `docs(modeling): specify legacy feature parity restoration`
- `feat(data): restore legacy model hierarchy features`
- `feat(modeling): select restored legacy price features`
- `feat(modeling): add catboost quantile interval helper`

Additional fixes discovered during implementation should use `fix(...)` commits and should not be bundled with unrelated work.

## Acceptance Criteria

- New Gold output contains the restored hierarchy, ordinal, interaction, and binary legacy features.
- Model feature selection includes the restored predictive columns.
- Representative rare-model examples map to the expected hierarchy categories.
- CatBoost quantile helper produces lower, median, and upper EUR predictions on a small dataset when CatBoost is installed.
- Existing focused benchmark/orchestration tests continue to pass.
- No unrelated notebook execution output or log files are included in commits.
