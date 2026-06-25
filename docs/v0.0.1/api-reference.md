# API Reference — EliminationSearchCV v0.0.1

Complete reference for the public API. Only public attributes and methods are documented here. Internal helpers (`_eliminate_single_param_values`, `_eliminate_multi_param_values`, `_score_candidates`) are implementation details and not part of the public API.

---

## Import

```python
from EliminationSearchCV import EliminationSearchCV
```

---

## Class: `EliminationSearchCV`

```
EliminationSearchCV(
    estimator,
    param_grid,
    scoring,
    cv=5,
    elimination_rate=0.8
)
```

A hyperparameter search strategy that progressively eliminates low-scoring parameter values across multiple rounds of cross-validated evaluation.

In each round, parameter combinations of increasing complexity are scored. Low-scoring values are pruned from the search grid based on `elimination_rate`, narrowing the search space before the next round begins.

---

### Constructor Parameters

| Parameter | Type | Default | Required | Description |
|---|---|---|---|---|
| `estimator` | sklearn estimator | — | ✅ | Any estimator implementing `fit` and `predict`. Must be sklearn-compatible (supports `clone()` and `set_params()`). |
| `param_grid` | `Dict[str, List]` | — | ✅ | Parameter names mapped to lists of candidate values. Same format as `GridSearchCV`. |
| `scoring` | `str` | — | ✅ | Metric name used to evaluate each candidate. See [Supported Scoring Metrics](#supported-scoring-metrics). |
| `cv` | `int` | `5` | ❌ | Number of cross-validation folds. Use `cv=1` for a single 80/20 train-validation split. |
| `elimination_rate` | `float` | `0.8` | ❌ | Fraction of low-scoring values to drop after each round. Must be in `[0.0, 1.0)`. `0.8` means keep the best 20%. |

#### Parameter Notes

**`estimator`**

The estimator is never modified in place. `EliminationSearchCV` always uses `sklearn.base.clone(estimator)` to create fresh copies before setting parameters and fitting. Your original estimator is untouched.

```python
base = LogisticRegression(random_state=42)
search = EliminationSearchCV(base, ...)
search.fit(X_train, y_train)
# base is still unfitted — it was never modified
```

**`param_grid`**

Supports any Python hashable values as parameter values. Incompatible combinations (e.g. `penalty='l1'` with `solver='lbfgs'`) are handled automatically — they receive a score of `0.0` and are eliminated naturally.

```python
param_grid = {
    'C':          [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty':    ['l1', 'l2'],
    'solver':     ['liblinear', 'saga'],
    'max_iter':   [100, 500, 1000],
    'fit_intercept': [True, False],
}
```

**`cv`**

Internally, `StratifiedKFold` is used by default (preserving class balance in each fold). For regression tasks or imbalanced datasets, this may not be appropriate — a future version will expose this as a parameter. For `cv=1`, a single 80/20 split with `random_state=42` is used.

**`elimination_rate`**

Controls how aggressively values are pruned each round:

| `elimination_rate` | Keep fraction | Behaviour |
|---|---|---|
| `0.0` | 100% | No elimination — equivalent to evaluating all combinations (still more work than `GridSearchCV`) |
| `0.5` | 50% | Moderate pruning. Recommended for small datasets or grids |
| `0.8` | 20% | Default. Aggressive pruning. Best for large grids with many values per parameter |
| `0.9` | 10% | Very aggressive. May prune good values on noisy datasets |

---

### Attributes

All attributes are **set to safe sentinel values before `fit()` is called** and are **populated after `fit()` completes**.

---

#### `best_params_`

**Type:** `Dict[str, Any]`  
**Sentinel (before fit):** `{}`  
**Set in:** `fit()`

The best hyperparameter combination found, as a flat dictionary of scalar values. All values are unwrapped from their list representation — they are ready to pass directly to `estimator.set_params(**best_params_)`.

```python
>>> search.best_params_
{'C': 1, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000}

# Equivalent to:
model = LogisticRegression()
model.set_params(**search.best_params_)
```

> This is the only attribute that existed in versions prior to v0.0.1.

---

#### `best_score_`

**Type:** `float`  
**Sentinel (before fit):** `0.0`  
**Set in:** `fit()`

The **mean cross-validated score** of `best_params_`, computed over the same CV folds that were built during the search. No extra data splits are performed — the existing folds are reused.

Mirrors the behaviour of `GridSearchCV.best_score_`.

```python
>>> search.best_score_
0.9248
```

> ⚠️ This is a cross-validation score on the **training set**, not a held-out test set. Use `best_estimator_.predict(X_test)` and score separately for an unbiased performance estimate.

---

#### `best_estimator_`

**Type:** sklearn estimator or `None`  
**Sentinel (before fit):** `None`  
**Set in:** `fit()`

A clone of `estimator`, configured with `best_params_`, and **re-fitted on the full training dataset** (all of `X` and `y` passed to `fit()`, not on fold subsets). It is ready to call `.predict()`, `.predict_proba()`, `.score()`, etc. on directly.

Mirrors the behaviour of `GridSearchCV.best_estimator_`.

```python
>>> search.best_estimator_
LogisticRegression(C=1, penalty='l1', solver='liblinear')

>>> search.best_estimator_.predict(X_test)
array([1, 0, 1, 1, 0, ...])

>>> search.best_estimator_.predict_proba(X_test[:2])
array([[0.12, 0.88],
       [0.76, 0.24]])
```

**When is it `None`?**  
If the refit on the full training dataset fails (e.g. the surviving `best_params_` combination is still incompatible with the estimator), `best_estimator_` is set to `None` instead of raising an exception. `best_params_` and `best_score_` remain valid in this case.

---

### Methods

---

#### `fit(X, y)` → `EliminationSearchCV`

Run the elimination search over `X` and `y`.

Iterates from single-parameter combinations (Round 1) up to full combinations of all parameters (Round `n_params`). After each round, low-scoring parameter values are pruned from the active grid. After all rounds, `best_params_`, `best_score_`, and `best_estimator_` are populated.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `X` | array-like of shape `(n_samples, n_features)` | Feature matrix. Accepts NumPy arrays and pandas DataFrames. |
| `y` | array-like of shape `(n_samples,)` | Target vector. Accepts NumPy arrays and pandas Series. |

**Returns:**

`self` — the fitted `EliminationSearchCV` instance. This follows sklearn's convention of returning `self` from `fit()`, allowing method chaining:

```python
search = EliminationSearchCV(...).fit(X_train, y_train)
```

**Raises:**

Does not raise on invalid parameter combinations — these are caught internally and scored `0.0`.

**Example:**

```python
search = EliminationSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid={
        'n_estimators':      [10, 50, 100, 300],
        'max_depth':         [None, 5, 10, 15],
        'min_samples_split': [2, 5, 10, 20],
        'criterion':         ['gini', 'entropy'],
    },
    scoring='accuracy',
    cv=5,
    elimination_rate=0.8,
)
search.fit(X_train, y_train)

print(search.best_params_)       # {'n_estimators': 50, 'max_depth': 5, ...}
print(search.best_score_)        # 0.9718
print(search.best_estimator_)    # RandomForestClassifier(...)
```

---

### Supported Scoring Metrics

Pass one of these strings as the `scoring` argument:

| `scoring` value | Sklearn function | Task |
|---|---|---|
| `'accuracy'` | `sklearn.metrics.accuracy_score` | Classification |
| `'precision'` | `sklearn.metrics.precision_score` | Binary classification |
| `'recall'` | `sklearn.metrics.recall_score` | Binary classification |
| `'f1'` | `sklearn.metrics.f1_score` | Binary classification |
| `'roc_auc'` | `sklearn.metrics.roc_auc_score` | Binary classification |

Passing an unsupported string raises `ValueError`:
```
ValueError: Unsupported scoring 'rmse'. Choose one of: ['accuracy', 'f1', 'precision', 'recall', 'roc_auc']
```

> **Custom callable scorers are not supported in v0.0.1.** This is planned for a future release. For regression or multi-class tasks, use `'accuracy'` or `GridSearchCV` with a custom scorer.

---

## Utility Functions (Advanced)

These functions are used internally by `EliminationSearchCV` but are importable independently for custom use.

### `generate_param_combinations_with_limit(param_grid, limit)`

```python
from EliminationSearchCV.Utils import generate_param_combinations_with_limit
```

Generates all combinations of exactly `limit` parameters at a time from `param_grid`.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `param_grid` | `Dict[str, List]` | Parameter grid |
| `limit` | `int` | Number of parameters to include in each combination. Must be in `[1, len(param_grid)]`. |

**Returns:** `List[Dict]` — list of parameter dicts, each with exactly `limit` keys.

```python
grid = {'C': [0.1, 1], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}

generate_param_combinations_with_limit(grid, limit=1)
# [{'C': 0.1}, {'C': 1}, {'penalty': 'l1'}, {'penalty': 'l2'}, {'solver': 'liblinear'}]

generate_param_combinations_with_limit(grid, limit=2)
# [{'C': 0.1, 'penalty': 'l1'}, {'C': 0.1, 'penalty': 'l2'}, ...]
```

---

### `create_cv_data_sets(X, y, cv=5, stratified=True)`

```python
from EliminationSearchCV.Utils import create_cv_data_sets
```

Splits data into cross-validation folds.

**Parameters:**

| Parameter | Type | Default | Description |
|---|---|---|---|
| `X` | array-like | — | Feature matrix |
| `y` | array-like | — | Target vector |
| `cv` | `int` | `5` | Number of folds. Use `1` for a single 80/20 split. |
| `stratified` | `bool` | `True` | If `True`, uses `StratifiedKFold`. Set `False` for regression. |

**Returns:** `List[Tuple]` — list of `(X_train, y_train, X_val, y_val)` tuples, one per fold.

---

### `get_model_score(model, X_val, y_val, scoring)`

```python
from EliminationSearchCV.Utils import get_model_score
```

Evaluates a fitted model on validation data using a single metric.

**Parameters:**

| Parameter | Type | Description |
|---|---|---|
| `model` | fitted estimator | A fitted sklearn estimator |
| `X_val` | array-like | Validation features |
| `y_val` | array-like | True labels |
| `scoring` | `str` | One of the [supported metric names](#supported-scoring-metrics) |

**Returns:** `float`

**Raises:** `ValueError` if `scoring` is not supported.
