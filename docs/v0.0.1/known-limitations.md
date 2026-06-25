# Known Limitations — EliminationSearchCV v0.0.1

This page documents honest, known limitations of the current implementation. Understanding these helps you decide when to use `EliminationSearchCV` and when to stick with `GridSearchCV`.

---

## Missing Features (Planned)

| Feature | Status | Impact | Workaround |
|---|---|---|---|
| `cv_results_` | ❌ Not implemented | Cannot inspect per-fold score breakdown | Use `best_score_` for a summary |
| `n_jobs` | ❌ Not implemented | Single-threaded only — slow on large grids | Reduce `cv` value or grid size |
| Custom callable `scoring` | ❌ Not implemented | Limited to 5 built-in metrics | Use `GridSearchCV` for custom scorers |
| `verbose` | ❌ Not implemented | No progress output during long searches | No workaround currently |
| `refit=False` flag | ❌ Not implemented | Cannot opt out of the refit on full data | Ignore `best_estimator_` and refit manually |
| `BaseEstimator` compatibility | ❌ Not implemented | `get_params()` / `set_params()` not available on the searcher itself | Not needed for basic use |
| PyPI publication | ❌ Not yet | Extra install step | Install from source (`pip install -e .`) |
| `random_state` constructor parameter | ❌ Not implemented | CV splits always use `random_state=42` internally | Set `random_state` on your estimator |

---

## Algorithm Limitations

### 1. Small Grids Run Slower Than GridSearchCV

**What happens:** For light/small parameter grids (< ~20 total combinations), `EliminationSearchCV` consistently runs **slower** than `GridSearchCV`.

**Why:** The algorithm runs `n_params` rounds of scoring regardless of grid size. On a tiny grid, this overhead exceeds the time saved by pruning.

**Benchmark evidence:**

| Model | Grid Size | Elim Time | Grid Time | Speedup |
|---|---|---|---|---|
| DecisionTree | Light | 0.06s | 0.03s | **0.6x** |
| RandomForest | Light | 1.15s | 0.35s | **0.3x** |
| GradientBoosting | Light | 2.64s | 0.39s | **0.1x** |

**Workaround:** Use `GridSearchCV` for small grids. `EliminationSearchCV` is designed for large grids (full benchmark shows 11x–152x speedup).

---

### 2. Parameter Interactions Are Not Modelled in Round 1

**What happens:** Round 1 scores each parameter value **in isolation**. If two parameters have strong interaction effects — their combined performance is much better than either individually suggests — the isolation scoring may prune good values before they get a chance to compete together.

**Example:** `GradientBoosting` Diabetes Full Grid shows a -0.0571 accuracy trade-off vs `GridSearchCV`. This is the worst case in v0.0.1 benchmarks and is caused by complex `n_estimators` + `learning_rate` + `subsample` interactions that Round 1 fails to detect.

**Workaround:** Lower `elimination_rate` (e.g. `0.5`) to retain more values through Round 1.

---

### 3. Aggressive `elimination_rate` on Small Datasets

**What happens:** On small datasets (< 500 rows), cross-validation score variance is high. A good parameter value may score poorly by chance in Round 1 and get eliminated permanently.

**Workaround:** Use `elimination_rate=0.5` or lower for small datasets.

---

### 4. `best_score_` Is Computed With an Extra Scoring Pass

**What happens:** After all elimination rounds complete, `best_params_` is re-scored over the existing CV folds to compute `best_score_`. This is one extra scoring pass that adds time for expensive models or large `cv` values.

**Planned fix:** Cache scores *during* the final round and retrieve the winner's score directly, eliminating the extra pass.

---

### 5. No Support for Multi-Metric Scoring

**What happens:** Only one `scoring` metric can be specified. There is no way to optimize for F1 but also track accuracy.

**Workaround:** Run the search with your primary metric, then evaluate `best_estimator_` against secondary metrics manually.

---

### 6. `StratifiedKFold` Is Always Used for CV (No Override)

**What happens:** `create_cv_data_sets` uses `StratifiedKFold` by default. For regression tasks, there is no way to override this through `EliminationSearchCV`'s constructor.

**Planned fix:** Expose `stratified` as a constructor parameter.

---

### 7. `random_state` for CV Splits Is Hardcoded

**What happens:** The CV fold splitter uses `random_state=42` internally. You cannot set a different random seed through the constructor.

**Planned fix:** Expose `random_state` as a constructor parameter.

---

## Edge Cases

### All Combinations Score `0.0`

If every parameter combination causes an exception during fitting, all candidates score `0.0`. `best_score_` will be `0.0` and `best_estimator_` may be `None`.

**Detection:** check `best_score_ == 0.0` after fitting as a warning signal.

---

### `param_grid` With a Single Parameter

Works correctly. Round 1 is the only round.

---

### `param_grid` With a Single Value Per Parameter

```python
param_grid = {'C': [1], 'penalty': ['l1']}
```

Works correctly. Single-value parameters are never eliminated. Result: `best_params_ = {'C': 1, 'penalty': 'l1'}`.

---

### `elimination_rate=0.0`

No values are eliminated. The full Cartesian product is evaluated but with the overhead of `n_params` rounds. **Use `GridSearchCV` instead.**

---

### `cv=1`

Supported. A single 80/20 train-validation split is used (`random_state=42`, stratified by default). Faster but less reliable.

---

## Reporting New Issues

If you encounter a bug or limitation not listed here, please [open an issue](https://github.com/thisal-d/elimination-search-cv/issues) with a minimal reproducible example.
