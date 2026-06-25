# EliminationSearchCV v0.0.1 — Documentation

> **Status:** Pre-Alpha — first working implementation. The public API is functional but may change without notice in future releases.

---

## What's in This Version

| Component | Status |
|---|---|
| `EliminationSearchCV` class | ✅ |
| `fit(X, y)` | ✅ |
| `best_params_` | ✅ |
| `best_score_` | ✅ |
| `best_estimator_` | ✅ |
| 5 scoring metrics (`accuracy`, `precision`, `recall`, `f1`, `roc_auc`) | ✅ |
| `StratifiedKFold` / `KFold` CV | ✅ |
| Invalid combination handling | ✅ |
| `cv_results_` | ❌ Not yet |
| `n_jobs` parallel execution | ❌ Not yet |
| Custom callable scorers | ❌ Not yet |
| PyPI package | ❌ Not yet |

---

## In This Documentation

| Page | Description |
|---|---|
| [Quickstart](./quickstart.md) | Get a working result in under 5 minutes |
| [API Reference](./api-reference.md) | Full `EliminationSearchCV` class reference |
| [How the Algorithm Works](./algorithm.md) | Elimination logic, rounds, worked examples |
| [Benchmarks](./benchmarks.md) | Speed and accuracy vs `GridSearchCV` (v0.0.1 results) |
| [Known Limitations](./known-limitations.md) | Honest list of current gaps and workarounds |

---

## Install (v0.0.1)

```bash
git clone https://github.com/thisal-d/elimination-search-cv.git
cd elimination-search-cv
git checkout v0.0.1          # pin to this version
pip install -e .
```

**Requirements:** Python ≥ 3.8, scikit-learn, numpy

---

## Minimal Example

```python
from EliminationSearchCV import EliminationSearchCV
from sklearn.linear_model import LogisticRegression

search = EliminationSearchCV(
    estimator=LogisticRegression(),
    param_grid={'C': [0.01, 0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear', 'saga']},
    scoring='accuracy',
    cv=5,
    elimination_rate=0.8,
)
search.fit(X_train, y_train)

print(search.best_params_)       # {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}
print(search.best_score_)        # 0.924
print(search.best_estimator_)    # LogisticRegression(C=1, penalty='l1', solver='liblinear')
```

→ [Full quickstart →](./quickstart.md)

---

*This version's documentation is frozen at release. For the latest version, see [docs/index.md](../index.md).*
