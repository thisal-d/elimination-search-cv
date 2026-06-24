# EliminationSearchCV 🚀

> A Scikit-Learn compatible hyperparameter search that **eliminates low-scoring parameter values progressively** — round by round — rather than blindly evaluating the entire Cartesian product up front.

[![Python](https://img.shields.io/badge/python-3.8%2B-blue)](https://www.python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](./LICENSE)
[![Status](https://img.shields.io/badge/status-pre--alpha-orange)](./CHANGELOG.md)
[![GitHub Issues](https://img.shields.io/github/issues/thisal-d/elimination-search-cv)](https://github.com/thisal-d/elimination-search-cv/issues)

If this project interests you, a ⭐ on the repo keeps the motivation alive — it genuinely helps!

---

## 💡 The Problem with `GridSearchCV`

Scikit-Learn's `GridSearchCV` is brute-force by design. For a grid with `k` parameters and `n` values each, it evaluates **nᵏ × cv_folds** configurations — every single one, regardless of how poorly a value performs in early trials.

| Problem | Impact |
|---|---|
| Dead-end values are never discarded | A bad `learning_rate=0.5` is re-evaluated in every downstream combination |
| No learning from early results | The search treats round 1 and round 1000 as equally uninformed |
| Exponential cost scaling | Adding one new 4-value parameter can **quadruple** total training time |

---

## ✅ The Elimination Approach

`EliminationSearchCV` evaluates parameters in **rounds of increasing complexity**, using cross-validated scores from each round to eliminate underperformers before they compound.

**Concrete example** — tuning `LogisticRegression` with:

```python
param_grid = {
    'C':        [0.001, 0.01, 0.1, 1, 10, 100],   # 6 values
    'penalty':  ['l1', 'l2'],                       # 2 values
    'solver':   ['liblinear', 'saga'],              # 2 values
    'max_iter': [1000, 2000],                       # 2 values
}
# Full GridSearchCV: 6 × 2 × 2 × 2 = 48 combinations × 5 folds = 240 fits
```

`EliminationSearchCV` with `reduce_rate=0.8` (keep best 20%):

| Round | Combinations tested | Grid after elimination |
|-------|--------------------|-----------------------|
| 1 — single-param | 12 | `C:[1], penalty:['l1'], solver:['liblinear'], max_iter:[1000]` |
| 2 — two-param pairs | 6 | unchanged (all at 1 value) |
| 3 — three-param triples | 4 | unchanged |
| 4 — full combinations | 1 | final result |
| **Total** | **23 fits** | vs **240 fits** for GridSearchCV (×5 folds) |

> ⚠️ **Note:** This is an experimental approach. The quality of the best result found — and how often it matches a full grid search — is actively being benchmarked. Results depend heavily on the dataset and model.

---

## 🏗️ Architecture & Component Breakdown

```
src/EliminationSearchCV/
├── EliminationSearchCV.py   ← Core class: fit(), elimination logic, scoring
└── Utils.py                 ← Stateless utilities: fold creation, combination generation, metrics
```

### Module Interaction

```
EliminationSearchCV.fit(X, y)
         │
         ├─▶ Utils.create_cv_data_sets()          — builds StratifiedKFold/KFold splits
         │
         └─▶ [For each round i = 1 … n_params]
                  │
                  ├─▶ Utils.generate_param_combinations_with_limit(grid, limit=i)
                  │         — generates all i-parameter combinations from active grid
                  │
                  ├─▶ EliminationSearchCV._score_candidates(candidates)
                  │         └─▶ Utils.get_model_score()  — per-fold metric evaluation
                  │
                  └─▶ EliminationSearchCV._eliminate_low_scoring_values(candidates, scores)
                            ├─▶ _eliminate_single_param_values()   — Round 1: per-param
                            └─▶ _eliminate_multi_param_values()    — Rounds 2+: global rank
```

### Key Design Decisions

| Decision | Rationale |
|---|---|
| **Per-parameter elimination in Round 1** | Each param is scored in isolation so its values are compared fairly, without interference from other params |
| **Global ranking in later rounds** | Multi-param combos are ranked by total cross-validated score; the top `(1-reduce_rate)` fraction survives |
| **Params not in any kept combo are preserved** | Prevents a parameter from being wiped out just because it wasn't part of the top-ranked 2-param pairs |
| **Invalid combos score `0.0`** | Incompatible combinations (e.g. `penalty='l1'` + `solver='lbfgs'`) are caught and naturally eliminated |
| **Always keep ≥ 1 value per param** | Prevents the grid from collapsing to an empty state |

---

## 📦 Installation

### From PyPI *(not yet published)*

```bash
pip install elimination-search-cv
```

### From Source (Developer Setup)

```bash
git clone https://github.com/thisal-d/elimination-search-cv.git
cd elimination-search-cv
pip install -e .
```

**Requirements:** Python ≥ 3.8, scikit-learn, numpy

---

## 🔌 Core API & Usage Examples

### Basic Usage

```python
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.datasets import make_classification

from EliminationSearchCV import EliminationSearchCV

# 1. Prepare data
X, y = make_classification(n_samples=5000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

# 2. Define model and grid (same format as GridSearchCV)
model = LogisticRegression(random_state=42)

param_grid = {
    'C':        [0.001, 0.01, 0.1, 1, 10, 100],
    'penalty':  ['l1', 'l2'],
    'solver':   ['liblinear', 'saga'],
    'max_iter': [1000, 2000],
}

# 3. Initialize and fit
search = EliminationSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
    cv=5,
    reduce_rate=0.8,   # eliminate worst 80% each round, keep best 20%
)
search.fit(X_train, y_train)

# 4. Use results — same interface as GridSearchCV
print(search.best_params_)
# → {'C': 1, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000}

best_model = LogisticRegression(**search.best_params_, random_state=42)
best_model.fit(X_train, y_train)
```

### Constructor Parameters

> **Current support only.** These are the parameters available right now. More options (e.g. `n_jobs`, `verbose`, `refit`) may or may not be added in the future — no promises yet.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimator` | sklearn estimator | *required* | Any estimator implementing `fit` and `predict`. |
| `param_grid` | `Dict[str, List]` | *required* | Parameter names mapped to candidate value lists. |
| `scoring` | `str` | *required* | Evaluation metric. See supported values below. |
| `cv` | `int` | `5` | Number of cross-validation folds. |
| `reduce_rate` | `float` | `0.8` | Fraction of values to eliminate per round. Must be in `[0.0, 1.0)`. |

### Result Attributes

> **Current support only.** Only `best_params_` is exposed after fitting right now. Attributes like `best_score_` and `cv_results_` are not yet available.

| Attribute | Type | Description |
|---|---|---|
| `best_params_` | `Dict[str, Any]` | Best parameter combination found, as scalar values. Ready to pass to `estimator.set_params(**best_params_)`. |

### Supported Scoring Metrics

> **Current support only.** These five metrics are what the library supports today. Additional metrics may be added later.

| Value | Sklearn function |
|---|---|
| `'accuracy'` | `sklearn.metrics.accuracy_score` |
| `'precision'` | `sklearn.metrics.precision_score` |
| `'recall'` | `sklearn.metrics.recall_score` |
| `'f1'` | `sklearn.metrics.f1_score` |
| `'roc_auc'` | `sklearn.metrics.roc_auc_score` |

---

## ⚡ GridSearchCV Comparison

### Strategy Differences

| | `GridSearchCV` | `EliminationSearchCV` |
|---|---|---|
| **Strategy** | Full Cartesian product | Progressive elimination |
| **Combinations evaluated** | `∏ len(values_i)` for all params | Shrinks each round as values are dropped |
| **Early stopping** | ✗ None | ✓ Low-scoring values dropped after Round 1 |
| **Invalid combo handling** | Raises exception | Scored `0.0`, eliminated naturally |
| **Params with 1 remaining value** | Not applicable | Skipped from further expansion (zero overhead) |

### 📊 Benchmarks

> **Experimental.** Results vary by dataset and hyperparameter grid. Run `python benchmarks/benchmark.py` to reproduce.
> Full detailed per-dataset tables → **[benchmark_results.md](./benchmarks/benchmark_results.md)**

**Settings:** `cv=5` · `reduce_rate=0.8` · `primary_scoring=accuracy` · `sample_size=5,000`  
**Models tested:** LogisticRegression · RandomForest · DecisionTree · KNeighbors · GradientBoosting  
**Metrics tracked:** Accuracy · F1 (macro) · Precision (macro) · Recall (macro) · Time (s) · Speedup  

#### 🚀 Speed & Score Summary (5-Model Run)

The table below shows average search times and accuracy differences compared to a full grid search across all 5 benchmark datasets.

| Model | Grid Combos | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| **LogisticRegression** | 40 | **0.57s** | 2.11s | **4.4x** | -0.0104 |
| **RandomForest** | 27 | **10.21s** | 10.14s | **1.0x** | -0.0060 |
| **DecisionTree** | 24 | **0.24s** | 0.17s | **0.8x** | -0.0017 |
| **KNeighbors** | 16 | **1.33s** | 0.45s | **0.4x** | -0.0035 |
| **GradientBoosting** | 18 | **11.86s** | 8.93s | **0.8x** | -0.0072 |

> **Key Findings:**
> 1. **High-dimensional grids (LogisticRegression)** show a **4.4x speedup** with minimal accuracy trade-off (~1%).
> 2. **Fast models (KNeighbors, DecisionTree)** have very low individual fit overhead, meaning the logic overhead of elimination doesn't pay off for small search spaces.
> 3. **Heavy models (RandomForest, GradientBoosting)** show comparable runtimes at this grid size, but we expect higher speedups on larger grids.
> 4. **Small datasets:** On very small datasets (e.g. Heart Failure, n=299), cross-validation score noise makes the 80% elimination rate too aggressive. We recommend using a lower `reduce_rate` (e.g., `0.5`) for datasets under 500 rows.

---


## 🔬 Internal Utilities (`Utils.py`)

These functions are used internally by `EliminationSearchCV` but are importable independently.

### `generate_param_combinations_with_limit(param_grid, limit)`

Generates all combinations of exactly `limit` parameters at a time.

```python
from EliminationSearchCV.Utils import generate_param_combinations_with_limit

grid = {'C': [0.1, 1], 'penalty': ['l1', 'l2']}

# limit=1: each param in isolation
generate_param_combinations_with_limit(grid, limit=1)
# → [{'C': 0.1}, {'C': 1}, {'penalty': 'l1'}, {'penalty': 'l2'}]

# limit=2: all pairs
generate_param_combinations_with_limit(grid, limit=2)
# → [{'C': 0.1, 'penalty': 'l1'}, {'C': 0.1, 'penalty': 'l2'},
#    {'C': 1,   'penalty': 'l1'}, {'C': 1,   'penalty': 'l2'}]
```

### `create_cv_data_sets(X, y, cv, stratified)`

Returns a list of `(X_train, y_train, X_val, y_val)` tuples — one per fold. Uses `StratifiedKFold` by default for classification, `KFold` when `stratified=False`.

### `get_model_score(model, X_val, y_val, scoring)`

Evaluates a fitted model against a single metric. Raises `ValueError` for unsupported metric names.

---

## 🚧 Project Status & Roadmap

### ✅ Implemented
- Core `EliminationSearchCV` class with `fit()`, `best_params_`
- Round 1: per-parameter isolation and elimination
- Rounds 2+: global combination ranking and elimination
- Cross-validated fold creation (`StratifiedKFold` / `KFold`)
- Invalid combination handling (score `0.0`)
- Scoring utilities for 5 metrics

### 🔨 In Progress / Planned
- `best_score_` and `cv_results_` attributes
- `n_jobs` parallel evaluation via `joblib`
- `verbose` logging parameter
- Scikit-Learn `BaseEstimator` compatibility (`get_params` / `set_params`)
- Full `pytest` test suite
- Systematic benchmarks vs `GridSearchCV` and `RandomizedSearchCV`
- PyPI publication (`v0.1.0`)
- Sphinx / MkDocs API documentation

---

## 🤝 Contributing

1. **Browse open issues** at [github.com/thisal-d/elimination-search-cv/issues](https://github.com/thisal-d/elimination-search-cv/issues)
2. **Open an issue** before starting significant work — aligns design, avoids duplicates
3. **Fork → feature branch → PR** against `main` (e.g. `feat/n-jobs-parallel`)
4. PRs should include tests and clean docstrings

---

## 📜 Changelog

| Version | Summary |
|---|---|
| [v0.0.1](./docs/changelogs/v0.0.1.md) | Initial working implementation: elimination rounds, cross-validated scoring, `best_params_`, invalid combo handling |

Full release history: [CHANGELOG.md](./CHANGELOG.md)

---

## License

MIT — see [LICENSE](./LICENSE).

---

<p align="center">
  Made with ❤️ by <a href="https://github.com/thisal-d">Thisal-D</a><br>
  <sub>If you find this project interesting or useful, please consider giving it a ⭐ — it really does help keep things moving.</sub>
</p>