# Quickstart — EliminationSearchCV v0.0.1

Get a working result in 5 minutes.

---

## 1. Install

```bash
git clone https://github.com/thisal-d/elimination-search-cv.git
cd elimination-search-cv
pip install -e .
```

**Requirements:** Python ≥ 3.8, scikit-learn, numpy

---

## 2. Minimal Working Example

```python
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split

from EliminationSearchCV import EliminationSearchCV

# --- Prepare data ---
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- Define model and parameter grid (same format as GridSearchCV) ---
param_grid = {
    'C':       [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty': ['l1', 'l2'],
    'solver':  ['liblinear', 'saga'],
}

# --- Search ---
search = EliminationSearchCV(
    estimator=LogisticRegression(random_state=42),
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    elimination_rate=0.8,    # eliminate worst 80% each round, keep best 20%
)
search.fit(X_train, y_train)

# --- Read results ---
print(search.best_params_)
# → {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}

print(search.best_score_)
# → 0.9248  (mean cross-validated accuracy of the best combination)

# best_estimator_ is already fitted on the full training set — no extra step needed
y_pred = search.best_estimator_.predict(X_test)
```

---

## 3. Understanding the Results

After calling `fit()`, three attributes are available:

### `best_params_`

A plain Python `dict` with the winning hyperparameter values as scalars:

```python
>>> search.best_params_
{'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
```

You can pass it directly to an estimator:

```python
model = LogisticRegression(**search.best_params_)
model.fit(X_train, y_train)
```

### `best_score_`

The **mean cross-validated score** of the best parameter combination, computed over the same CV folds used during the search. This is the same value that `GridSearchCV.best_score_` returns.

```python
>>> search.best_score_
0.9248
```

> ⚠️ This is a cross-validation score on the training set, not a test-set score. Always evaluate on a held-out test set for an unbiased performance estimate.

### `best_estimator_`

A **pre-fitted** clone of your estimator, configured with `best_params_` and trained on the **full training dataset** (all of `X_train`, not fold subsets). It is ready to call `.predict()` on immediately.

```python
>>> search.best_estimator_
LogisticRegression(C=1, penalty='l1', solver='liblinear')

>>> search.best_estimator_.predict(X_test[:5])
array([1, 0, 1, 1, 0])
```

> If the best parameters turn out to be incompatible with the estimator at refit time (an edge case — see [Known Limitations](./known-limitations.md)), `best_estimator_` will be `None`. `best_params_` and `best_score_` are always valid.

---

## 4. EliminationSearchCV vs GridSearchCV — When to Use Which

| Situation | Recommendation |
|---|---|
| **Large grid** (many parameters, many values per param) | ✅ Use `EliminationSearchCV` — speedup can reach 30–150x |
| **Small grid** (< ~20 total combinations) | ⚠️ Use `GridSearchCV` — elimination overhead doesn't pay off |
| **Small dataset** (< 500 rows) | ⚠️ Lower `elimination_rate` to `0.5` or use `GridSearchCV` — CV noise may prune good values early |
| **You need `cv_results_`** or per-fold diagnostics | ❌ Use `GridSearchCV` — `cv_results_` is not yet implemented |
| **Custom scoring function** | ❌ Use `GridSearchCV` — only string metrics supported in v0.0.1 |

---

## 5. Common Pitfalls

### "My results differ from GridSearchCV"

This is expected and by design. Elimination prunes the grid early, so it may not explore every combination that GridSearchCV would. The benchmark results show that in most cases the score difference is < 0.002, but it can be larger for complex models on noisy datasets. See [Benchmarks](./benchmarks.md).

### "EliminationSearchCV is slower than GridSearchCV"

If your grid has fewer than ~20 total combinations, the overhead of running `n_params` rounds of scoring outweighs the savings from elimination. EliminationSearchCV is designed for **large** grids. Use a smaller `cv` value (e.g. `cv=2`) to speed up the search, or simply use `GridSearchCV` for small grids.

### "I got `best_estimator_ = None`"

This means the winning parameters from elimination are incompatible with the estimator at refit time (e.g. an invalid `solver`/`penalty` combination that somehow survived). `best_params_` and `best_score_` are still valid — you can manually create and fit the estimator using `best_params_`.

### "I'm getting different results on every run"

The cross-validation splits use `random_state=42` internally (hardcoded in v0.0.1). If you are getting different results, check whether your estimator itself uses randomness (e.g. `RandomForestClassifier` without a fixed `random_state`). Set `random_state` on the estimator passed to `EliminationSearchCV`.

---

## Next Steps

- [API Reference](./api-reference.md) — full parameter and attribute documentation
- [How the Algorithm Works](./algorithm.md) — understand what elimination is actually doing
- [Benchmarks](./benchmarks.md) — see real speed and accuracy numbers
