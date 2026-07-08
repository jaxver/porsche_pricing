# Legacy Feature Modeling Parity Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Restore legacy notebook feature engineering and reusable modeling behavior in the production Python pipeline.

**Architecture:** Keep the current Bronze -> Silver -> Gold -> Modeling package structure. Add deterministic legacy feature helpers to `silver_to_gold.py`, expand model feature selection in `features.py`, and add CatBoost quantile interval helpers in `catboost_model.py` without rewriting the existing benchmark orchestration.

**Tech Stack:** Python 3.13, pandas, numpy, scikit-learn, CatBoost, pytest, existing `rtk pytest` workflow with explicit `--basetemp`.

---

## File Structure

- Modify `elferspot_listings/data_processing/silver_to_gold.py`: owns Gold-layer feature engineering, including legacy model hierarchy, ordinal category codes, mileage/category interactions, and normalized binary flags.
- Modify `elferspot_listings/modeling/features.py`: owns benchmark/model feature allowlists and must select restored legacy columns when present.
- Modify `elferspot_listings/modeling/catboost_model.py`: owns CatBoost training/prediction helpers and will expose quantile interval training.
- Modify `tests/test_silver_to_gold.py`: locks in rare hierarchy and interaction feature formulas.
- Modify `tests/test_modeling_features.py`: locks in restored feature selection.
- Modify or create `tests/test_catboost_config.py`: adds CatBoost quantile helper smoke coverage with `pytest.importorskip("catboost")`.

Commits must be atomic and semantic. Do not stage executed notebooks, logs, benchmark artifacts, or unrelated uncommitted files.

---

### Task 1: Restore Legacy Model Hierarchy

**Files:**
- Modify: `elferspot_listings/data_processing/silver_to_gold.py`
- Test: `tests/test_silver_to_gold.py`

- [ ] **Step 1: Add failing tests for rare model hierarchy**

Append these tests to `tests/test_silver_to_gold.py`:

```python
def test_create_model_categories_restores_legacy_hierarchy():
    df = pd.DataFrame(
        {
            "Model": [
                "Porsche 911 GT2 RS",
                "Porsche 911 GT3 RS",
                "Porsche 964 Carrera RS",
                "Porsche 911 Speedster",
                "Singer 911",
                "Porsche 911 Turbo S",
                "Porsche 911 GTS",
                "Porsche 911 Carrera 3.2",
                "Porsche 912 Coupe",
                "Porsche Cayman GT4",
                "Porsche Boxster",
            ]
        }
    )

    result = create_model_categories(df)

    assert result["model_category"].tolist() == [
        "GT2RS and RARE Models",
        "GT3RS",
        "RS Model",
        "Special / Backdate",
        "Bespoke / Rarest Models",
        "Turbo S / Turbo",
        "GTS",
        "Carrera 3.0/3.2 / S / SC",
        "Base Carrera / Targa / 912",
        "GT4 / GT3 / GT2",
        "718",
    ]


def test_create_model_categories_prefers_specific_match_over_generic_911():
    df = pd.DataFrame({"Model": ["Porsche 911 Carrera RS", "Porsche 911 Carrera"]})

    result = create_model_categories(df)

    assert result["model_category"].tolist() == ["RS Model", "Base Carrera / Targa / 912"]
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "model_categories" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: new hierarchy tests fail because the current mapper returns `911`, `718`, or `Other`.

- [ ] **Step 3: Implement legacy hierarchy constants and categorizer**

In `elferspot_listings/data_processing/silver_to_gold.py`, replace the current simplified `model_mapping` implementation in `create_model_categories` with these module-level constants and helper:

```python
MODEL_CATEGORY_RULES: tuple[tuple[str, str], ...] = (
    (r"\b(singer|guntherwerks|gunther werks|lanzante)\b", "Bespoke / Rarest Models"),
    (
        r"(gt2 rs|rsr|gt2 rsr|911 gt2 rs|sport classic|911 st\b|911 s[\s/]?t|60 (jahre|years|anniversary)|911 r\b|le mans centenaire edition|991 club coup[eé]|club coup[eé])",
        "GT2RS and RARE Models",
    ),
    (r"\b(gt3 rs|gt3rs|911 gt3 rs|ruf|dakar|gt2 clubsport)\b", "GT3RS"),
    (
        r"\b(964 carrera rs|993 carrera rs|carrera rs\b|911 carrera rs\b|rs america|911 carrera 2\.7|911 carrera 2,7|911 carrera 2\.7 rs|911 carrera 2\.7 mfi|flachbau|gt4 rs|gt4rs|leichtbau)\b",
        "RS Model",
    ),
    (r"\b(gt3\b(?! rs)|gt2\b(?! rs)|911 gt3\b(?! rs)|911 gt2\b(?! rs)|cup|gt4|911 carrera 3\.2 clubsport)\b", "GT4 / GT3 / GT2"),
    (r"\b(speedster|clubsport|heritage|backdate|restomod|modified|exclusive manufaktur)\b", "Special / Backdate"),
    (r"\b(turbo s|turbo|930)\b", "Turbo S / Turbo"),
    (r"\b(gts)\b", "GTS"),
    (r"\b(carrera 3\.0|carrera 3,0|carrera 3\.2|carrera 3,2|911 sc|super carrera|carrera s|carrera 4s)\b", "Carrera 3.0/3.2 / S / SC"),
    (r"\b(912\b|911\b|911 t\b|911 l\b|911 e\b|911 targa\b|carrera\b|carrera 2\b|cabriolet|targa|coupe|convertible)\b", "Base Carrera / Targa / 912"),
    (r"\b(boxster|cayman|718|981|982|987)\b", "718"),
)

MODEL_CATEGORY_ORDER: tuple[str, ...] = (
    "Base Carrera / Targa / 912",
    "Carrera 3.0/3.2 / S / SC",
    "GTS",
    "Turbo S / Turbo",
    "GT4 / GT3 / GT2",
    "Special / Backdate",
    "RS Model",
    "GT3RS",
    "GT2RS and RARE Models",
    "Bespoke / Rarest Models",
    "718",
    "Other",
)
```

Then update `create_model_categories`:

```python
def create_model_categories(df: pd.DataFrame) -> pd.DataFrame:
    """Create legacy model hierarchy categories for better price generalization."""
    logger.info("Creating model categories")

    if "Model" not in df.columns:
        logger.warning("Model column not found, skipping categorization")
        return df

    def categorize_model(model: object) -> str:
        if pd.isna(model):
            return "Other"
        model_lower = str(model).lower().strip()
        for pattern, category in MODEL_CATEGORY_RULES:
            if re.search(pattern, model_lower, flags=re.IGNORECASE):
                return category
        return "Other"

    df = df.copy()
    df["model_category"] = df["Model"].apply(categorize_model)
    logger.info("Created %s model categories", df["model_category"].nunique())
    return df
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "model_categories" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: hierarchy tests pass.

- [ ] **Step 5: Commit**

Run:

```powershell
rtk git add elferspot_listings/data_processing/silver_to_gold.py tests/test_silver_to_gold.py
rtk git commit -m "feat(data): restore legacy model hierarchy"
```

---

### Task 2: Restore Legacy Gold Interaction Features

**Files:**
- Modify: `elferspot_listings/data_processing/silver_to_gold.py`
- Test: `tests/test_silver_to_gold.py`

- [ ] **Step 1: Add failing tests for ordinal and interaction features**

Append these tests to `tests/test_silver_to_gold.py`:

```python
def test_add_legacy_model_interaction_features_uses_ordered_category_code():
    df = pd.DataFrame(
        {
            "Model": ["Porsche 911 Carrera", "Porsche 911 GT2 RS"],
            "Mileage_km": [10000.0, 20000.0],
        }
    )
    df = create_log_features(df)
    df = create_model_categories(df)

    result = add_legacy_model_interaction_features(df)

    assert result["model_cat_ordered"].tolist() == [0, 8]
    assert result["inv_mileage"].tolist() == [1 / 10001.0, 1 / 20001.0]
    assert result["Mileage_model_cat"].tolist() == [0.0, 160000.0]
    assert result["inv_Mileage_model_cat"].tolist() == [0.0, 8 / 20001.0]
    assert result["Mileage_sq_model_cat"].tolist() == [0.0, (20000.0**2) * 8]


def test_add_legacy_model_interaction_features_handles_missing_mileage():
    df = pd.DataFrame({"Model": ["Unknown"], "Mileage_km": [pd.NA]})
    df = create_log_features(df)
    df = create_model_categories(df)

    result = add_legacy_model_interaction_features(df)

    assert result["model_cat_ordered"].tolist() == [11]
    assert pd.isna(result.loc[0, "Mileage_model_cat"])
```

- [ ] **Step 2: Run tests and verify they fail**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "legacy_model_interaction" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: fail because `add_legacy_model_interaction_features` does not exist.

- [ ] **Step 3: Implement interaction helper**

Add this function to `elferspot_listings/data_processing/silver_to_gold.py` after `create_model_categories`:

```python
def add_legacy_model_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy ordered model category and mileage interaction features."""
    if "model_category" not in df.columns:
        df = create_model_categories(df)

    result = df.copy()
    categories = pd.Categorical(result["model_category"], categories=MODEL_CATEGORY_ORDER, ordered=True)
    codes = pd.Series(categories.codes, index=result.index).replace(-1, len(MODEL_CATEGORY_ORDER) - 1)
    result["model_cat_ordered"] = codes.astype(int)

    mileage = pd.to_numeric(result.get("Mileage_km"), errors="coerce")
    if "Mileage_sq" not in result.columns:
        result["Mileage_sq"] = mileage**2
    mileage_sq = pd.to_numeric(result["Mileage_sq"], errors="coerce")

    result["inv_mileage"] = 1 / (mileage + 1)
    result["Mileage_model_cat"] = mileage * result["model_cat_ordered"]
    result["inv_Mileage_model_cat"] = result["inv_mileage"] * result["model_cat_ordered"]
    result["Mileage_sq_model_cat"] = mileage_sq * result["model_cat_ordered"]
    return result
```

In `process_silver_to_gold`, call it immediately after `create_model_categories(df)`:

```python
df = create_model_categories(df)
df = add_legacy_model_interaction_features(df)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "legacy_model_interaction or model_categories" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
rtk git add elferspot_listings/data_processing/silver_to_gold.py tests/test_silver_to_gold.py
rtk git commit -m "feat(data): add legacy model interaction features"
```

---

### Task 3: Restore Legacy Binary Modeling Flags

**Files:**
- Modify: `elferspot_listings/data_processing/silver_to_gold.py`
- Test: `tests/test_silver_to_gold.py`

- [ ] **Step 1: Add failing tests for binary flags**

Append this test to `tests/test_silver_to_gold.py`:

```python
def test_add_legacy_binary_flags_normalizes_ready_drive_and_matching_numbers():
    df = pd.DataFrame(
        {
            "Ready to drive": ["Yes", "no", ""],
            "Drive": ["Rear drive", "All wheel drive", "RWD"],
            "Matching numbers": ["Yes", "Unknown", "matching numbers"],
        }
    )

    result = add_legacy_binary_flags(df)

    assert result["state_yes"].tolist() == [1, 0, 0]
    assert result["state_Rear drive"].tolist() == [1, 0, 1]
    assert result["matching_yes"].tolist() == [1, 0, 1]
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "legacy_binary_flags" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: fail because `add_legacy_binary_flags` does not exist.

- [ ] **Step 3: Implement binary flag helper**

Add this function to `silver_to_gold.py`:

```python
def add_legacy_binary_flags(df: pd.DataFrame) -> pd.DataFrame:
    """Add legacy normalized binary features used by historical modeling notebooks."""
    result = df.copy()

    ready = result.get("Ready to drive", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    drive = result.get("Drive", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()
    matching = result.get("Matching numbers", pd.Series("", index=result.index)).fillna("").astype(str).str.strip().str.lower()

    result["state_yes"] = ready.isin({"yes", "y", "true", "1"}).astype(int)
    result["state_Rear drive"] = drive.str.contains(r"\b(rear|rwd)\b", regex=True).astype(int)
    result["matching_yes"] = matching.str.contains(r"\b(yes|matching numbers|numbers matching|matching drivetrain)\b", regex=True).astype(int)
    return result
```

In `process_silver_to_gold`, call it before `prepare_modeling_features(df)`:

```python
df = calculate_listing_score(df)
df = add_legacy_binary_flags(df)
df = prepare_modeling_features(df)
```

- [ ] **Step 4: Run tests and verify they pass**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py -k "legacy_binary_flags or legacy_model_interaction or model_categories" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
rtk git add elferspot_listings/data_processing/silver_to_gold.py tests/test_silver_to_gold.py
rtk git commit -m "feat(data): add legacy binary modeling flags"
```

---

### Task 4: Select Restored Legacy Features For Models

**Files:**
- Modify: `elferspot_listings/modeling/features.py`
- Test: `tests/test_modeling_features.py`

- [ ] **Step 1: Add failing feature selection test**

Append this test to `tests/test_modeling_features.py`:

```python
def test_select_model_columns_includes_restored_legacy_features():
    df = pd.DataFrame(
        {
            "price_in_eur": [100000.0],
            "Mileage_km": [10000.0],
            "model_cat_ordered": [8],
            "inv_mileage": [1 / 10001.0],
            "Mileage_model_cat": [80000.0],
            "inv_Mileage_model_cat": [8 / 10001.0],
            "Mileage_sq_model_cat": [800000000.0],
            "is_rare": [1],
            "is_restomod": [0],
            "is_race_ready": [1],
            "restoration_full": [0],
            "has_docs": [1],
            "state_yes": [1],
            "state_Rear drive": [1],
            "matching_yes": [1],
            "Model": ["Porsche 911 GT2 RS"],
            "model_category": ["GT2RS and RARE Models"],
        }
    )

    selected = select_model_columns(df)

    for column in (
        "model_cat_ordered",
        "inv_mileage",
        "Mileage_model_cat",
        "inv_Mileage_model_cat",
        "Mileage_sq_model_cat",
        "is_rare",
        "is_restomod",
        "is_race_ready",
        "restoration_full",
        "has_docs",
        "state_yes",
        "state_Rear drive",
        "matching_yes",
    ):
        assert column in selected.numeric
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
rtk pytest tests/test_modeling_features.py -k "restored_legacy_features" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: fail because allowlists do not include restored columns.

- [ ] **Step 3: Extend numeric allowlist**

In `elferspot_listings/modeling/features.py`, extend `NUMERIC_ALLOWLIST` to include:

```python
    "model_cat_ordered",
    "inv_mileage",
    "Mileage_model_cat",
    "inv_Mileage_model_cat",
    "Mileage_sq_model_cat",
    "restoration_full",
    "restoration_partial",
    "is_restomod",
    "has_docs",
    "is_matching_numbers",
    "is_mint",
    "is_race_ready",
    "is_rare",
    "is_accident_free",
    "has_upgrades",
    "first_owner",
    "state_yes",
    "state_Rear drive",
    "matching_yes",
```

- [ ] **Step 4: Run test and verify it passes**

Run:

```powershell
rtk pytest tests/test_modeling_features.py -k "restored_legacy_features" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: pass.

- [ ] **Step 5: Commit**

Run:

```powershell
rtk git add elferspot_listings/modeling/features.py tests/test_modeling_features.py
rtk git commit -m "feat(modeling): select restored legacy price features"
```

---

### Task 5: Add CatBoost Quantile Interval Helper

**Files:**
- Modify: `elferspot_listings/modeling/catboost_model.py`
- Test: `tests/test_catboost_config.py`

- [ ] **Step 1: Add failing CatBoost quantile smoke test**

Append this test to `tests/test_catboost_config.py`:

```python
def test_fit_catboost_quantile_interval_returns_eur_bounds():
    pytest.importorskip("catboost")
    from elferspot_listings.modeling.catboost_model import fit_catboost_quantile_interval, predict_catboost_interval_eur
    from elferspot_listings.modeling.features import select_model_columns

    train = pd.DataFrame(
        {
            "price_in_eur": [80000.0, 90000.0, 120000.0, 150000.0, 250000.0, 300000.0],
            "Mileage_km": [90000.0, 70000.0, 50000.0, 30000.0, 15000.0, 10000.0],
            "model_category": ["Base Carrera / Targa / 912", "Base Carrera / Targa / 912", "GTS", "Turbo S / Turbo", "RS Model", "GT2RS and RARE Models"],
        }
    )
    selected = select_model_columns(train)
    X = train[list(selected.features)]
    y = train["price_in_eur"]

    interval = fit_catboost_quantile_interval(
        X,
        y,
        selected,
        random_state=42,
        params={"iterations": 5, "depth": 2, "learning_rate": 0.1, "allow_writing_files": False, "verbose": False},
    )
    predictions = predict_catboost_interval_eur(interval, X)

    assert list(predictions.columns) == ["pred_lower", "pred_price", "pred_upper"]
    assert len(predictions) == len(train)
    assert (predictions["pred_lower"] <= predictions["pred_price"]).all()
    assert (predictions["pred_price"] <= predictions["pred_upper"]).all()
```

- [ ] **Step 2: Run test and verify it fails**

Run:

```powershell
rtk pytest tests/test_catboost_config.py -k "quantile_interval" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: fail because `fit_catboost_quantile_interval` and `predict_catboost_interval_eur` do not exist.

- [ ] **Step 3: Implement quantile interval helper**

Add this to `elferspot_listings/modeling/catboost_model.py`:

```python
def _prepare_catboost_frame(X, selected: SelectedColumns) -> tuple[pd.DataFrame, list[int]]:
    frame = pd.DataFrame(X).copy()
    categorical_columns = [column for column in selected.categorical if column in frame.columns]
    for col in categorical_columns:
        frame[col] = frame[col].fillna("Unknown").astype(str)
    cat_features = [frame.columns.get_loc(column) for column in categorical_columns]
    return frame, cat_features


def fit_catboost_quantile_interval(
    X_train,
    y_train,
    selected: SelectedColumns,
    random_state: int = 42,
    params: dict[str, Any] | None = None,
    device: str = "cpu",
    gpu_devices: str | None = None,
) -> dict[str, Any]:
    """Fit lower, median, and upper CatBoost quantile models on log price."""
    from catboost import CatBoostRegressor, Pool

    frame, cat_features = _prepare_catboost_frame(X_train, selected)
    target = np.asarray(y_train, dtype=float)
    if np.any(target <= 0):
        raise ValueError("Target values must be positive before applying the log transform")

    train_pool = Pool(frame, label=np.log(target), cat_features=cat_features)
    base_params = default_catboost_params(random_state=random_state)
    if params:
        base_params.update(params)
    base_params.update(_gpu_catboost_params(device=device, gpu_devices=gpu_devices))

    fitted: dict[str, Any] = {}
    for name, alpha in {"lower": 0.05, "median": 0.5, "upper": 0.95}.items():
        model_params = {**base_params, "loss_function": f"Quantile:alpha={alpha}"}
        model = CatBoostRegressor(**model_params)
        model.fit(train_pool)
        fitted[name] = model
    return fitted


def predict_catboost_interval_eur(interval_models: dict[str, Any], X) -> pd.DataFrame:
    """Predict lower, median, and upper CatBoost interval bounds in EUR."""
    frame = pd.DataFrame(X).copy()
    predictions = pd.DataFrame(index=frame.index)
    predictions["pred_lower"] = np.exp(np.asarray(interval_models["lower"].predict(frame), dtype=float))
    predictions["pred_price"] = np.exp(np.asarray(interval_models["median"].predict(frame), dtype=float))
    predictions["pred_upper"] = np.exp(np.asarray(interval_models["upper"].predict(frame), dtype=float))
    ordered = np.sort(predictions[["pred_lower", "pred_price", "pred_upper"]].to_numpy(dtype=float), axis=1)
    predictions.loc[:, ["pred_lower", "pred_price", "pred_upper"]] = ordered
    return predictions
```

Then update `fit_catboost_regressor` to use `_prepare_catboost_frame` to remove duplicate categorical preparation:

```python
frame, cat_features = _prepare_catboost_frame(X_train, selected)
```

- [ ] **Step 4: Run quantile test and focused CatBoost tests**

Run:

```powershell
rtk pytest tests/test_catboost_config.py -k "quantile_interval or catboost" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: pass or skip only when CatBoost is unavailable.

- [ ] **Step 5: Commit**

Run:

```powershell
rtk git add elferspot_listings/modeling/catboost_model.py tests/test_catboost_config.py
rtk git commit -m "feat(modeling): add catboost quantile interval helper"
```

---

### Task 6: Pipeline Verification And Final Guardrails

**Files:**
- Modify only if a preceding test reveals a bug.

- [ ] **Step 1: Run focused restored-feature tests**

Run:

```powershell
rtk pytest tests/test_silver_to_gold.py tests/test_modeling_features.py tests/test_catboost_config.py -k "model_categories or legacy_model_interaction or legacy_binary_flags or restored_legacy_features or quantile_interval" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: all selected tests pass, except CatBoost-specific test may skip if CatBoost is unavailable.

- [ ] **Step 2: Run existing benchmark orchestration checks**

Run:

```powershell
rtk pytest tests/test_train_baselines.py -k "attempts_catboost_before_heavy_optionals or autogluon_training_failure or catboost_training_failure or benchmark_db_is_redirected" --basetemp="C:\Users\jaxon\AppData\Local\Temp\opencode\pytest"
```

Expected: `4 passed`.

- [ ] **Step 3: Regenerate Silver and Gold for smoke verification**

Run:

```powershell
& "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m elferspot_listings.data_processing.bronze_to_silver
& "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -m elferspot_listings.data_processing.silver_to_gold
```

Expected: commands complete and `data/gold/all_listings_gold.xlsx` contains restored columns.

- [ ] **Step 4: Verify restored columns exist in Gold**

Run:

```powershell
& "C:\Users\jaxon\.venvs\Elferspot_prod\Scripts\python.exe" -c "import pandas as pd; df=pd.read_excel('data/gold/all_listings_gold.xlsx'); cols=['model_category','model_cat_ordered','inv_mileage','Mileage_model_cat','inv_Mileage_model_cat','Mileage_sq_model_cat','is_rare','is_restomod','is_race_ready','state_yes','state_Rear drive','matching_yes']; missing=[c for c in cols if c not in df.columns]; print('missing=', missing); print(df['model_category'].value_counts().head(12).to_string())"
```

Expected: `missing= []` and rare model categories appear when present in the data.

- [ ] **Step 5: Commit regenerated data only if explicitly requested**

Data files are gitignored. Do not commit generated Excel files, notebook execution output, benchmark output, or logs.

- [ ] **Step 6: Final status check**

Run:

```powershell
rtk git status --short
```

Expected: only unrelated pre-existing files remain unstaged, or the working tree contains exactly the files intentionally left uncommitted.
