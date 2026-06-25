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

`EliminationSearchCV` with `elimination_rate=0.8` (keep best 20%):

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
| **Global ranking in later rounds** | Multi-param combos are ranked by total cross-validated score; the top `(1-elimination_rate)` fraction survives |
| **Params not in any kept combo are preserved** | Prevents a parameter from being wiped out just because it wasn't part of the top-ranked 2-param pairs |
| **Invalid combos score `0.0`** | Incompatible combinations (e.g. `penalty='l1'` + `solver='lbfgs'`) are caught and naturally eliminated |
| **Always keep ≥ 1 value per param** | Prevents the grid from collapsing to an empty state |

---

## 📦 Installation

```bash
pip install elimination-search-cv
```

**Requirements:** Python ≥ 3.8 · `scikit-learn` and `numpy` are installed automatically.

### Developer Setup (contributing / running from source)

```bash
git clone https://github.com/thisal-d/elimination-search-cv.git
cd elimination-search-cv
pip install -e .
```

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
    elimination_rate=0.8,   # eliminate worst 80% each round, keep best 20%
)
search.fit(X_train, y_train)

# 4. Use results — same interface as GridSearchCV
print(search.best_params_)
# → {'C': 1, 'penalty': 'l1', 'solver': 'liblinear', 'max_iter': 1000}

print(search.best_score_)
# → 0.9248   (mean CV accuracy of the best combination)

# best_estimator_ is already fitted on the full training set — ready to predict
print(search.best_estimator_.predict(X_test[:5]))
# → [1 0 1 1 0]
```

### Constructor Parameters

> **Current support only.** These are the parameters available right now. More options (e.g. `n_jobs`, `verbose`, `refit`) may or may not be added in the future — no promises yet.

| Parameter | Type | Default | Description |
|---|---|---|---|
| `estimator` | sklearn estimator | *required* | Any estimator implementing `fit` and `predict`. |
| `param_grid` | `Dict[str, List]` | *required* | Parameter names mapped to candidate value lists. |
| `scoring` | `str` | *required* | Evaluation metric. See supported values below. |
| `cv` | `int` | `5` | Number of cross-validation folds. |
| `elimination_rate` | `float` | `0.8` | Fraction of values to eliminate per round. Must be in `[0.0, 1.0)`. |

### Result Attributes

| Attribute | Type | Description |
|---|---|---|
| `best_params_` | `Dict[str, Any]` | Best parameter combination found, as scalar values. Ready to pass to `estimator.set_params(**best_params_)`. |
| `best_score_` | `float` | Mean cross-validated score of the best parameter combination (same CV folds used during the search). Mirrors `GridSearchCV.best_score_`. |
| `best_estimator_` | sklearn estimator | A clone of `estimator` configured with `best_params_` and **re-fitted on the full training dataset** — ready to call `.predict()` directly. `None` if the refit fails. |

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

> **Experimental.** Results vary by dataset and hyperparameter grid.
> - Run the quick, configurable benchmark: `python benchmarks/benchmark_fast.py`
> - Run the comprehensive full benchmark: `python benchmarks/benchmark.py`
> - Full per-model, per-dataset tables (Light grid vs Full grid) → **[v0.0.1 benchmark results](./benchmarks/marks/benchmark_results_scaling_cv2_rate08_size10k_v0.0.1.md)**

**Settings:** `cv=2` · `elimination_rate=0.8` · `primary_scoring=accuracy` · `sample_size=10,000`  
**Models tested:** LogisticRegression · RandomForest · DecisionTree · KNeighbors · GradientBoosting  
**Reproduced with:** `python benchmarks/benchmark.py`  

#### 🚀 Speed & Score Summary — Light Grid vs Full Grid (v0.0.1)

The table below shows average search times and accuracy differences vs a full `GridSearchCV` across 3 benchmark datasets.

| Model | Grid Size | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| DecisionTree | Light | **0.06s** | 0.03s | **0.6x** | -0.0001 |
| **DecisionTree** | **Full** | **0.65s** | 81.48s | **152.5x** | -0.0008 |
| GradientBoosting | Light | **2.64s** | 0.39s | **0.1x** | +0.0000 |
| **GradientBoosting** | **Full** | **39.46s** | 1408.66s | **35.5x** | -0.0194 |
| KNeighbors | Light | **0.56s** | 0.13s | **0.3x** | +0.0000 |
| **KNeighbors** | **Full** | **8.77s** | 102.31s | **11.4x** | -0.0004 |
| LogisticRegression | Light | **0.11s** | 1.44s | **5.8x** | -0.0004 |
| **LogisticRegression** | **Full** | **1.10s** | 4.54s | **4.0x** | -0.0004 |
| RandomForest | Light | **1.15s** | 0.35s | **0.3x** | +0.0000 |
| **RandomForest** | **Full** | **33.58s** | 950.79s | **36.2x** | -0.0002 |

> **Key Findings:**
> 1. **Full grids are where elimination shines.** `DecisionTree` achieves a **152x speedup** on full grids with near-identical accuracy (-0.0008). `RandomForest` reaches **36x** and `GradientBoosting` **35x**.
> 2. **Light grids (small search spaces)** show slower-than-GridSearchCV times — the overhead of elimination rounds doesn't pay off when there are few combinations to begin with. This is expected behaviour.
> 3. **Score trade-off is minimal.** Across all models and datasets, the average accuracy difference on full grids is < 0.02, often zero.
> 4. **Small datasets / Light grids:** Use a lower `elimination_rate` (e.g., `0.5`) when the grid is small or the dataset is under 500 rows to avoid over-aggressive pruning.

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

> ⚠️ **Early-stage project.** `EliminationSearchCV` is functional but still at an early stage — the algorithm, API, and supported features will evolve significantly.
> More scorers, parallel execution (`n_jobs`), and richer result attributes are on the roadmap.
> **Read the [CHANGELOG](./CHANGELOG.md)** to follow what changes between versions, and **watch / ⭐ the repo** to be notified of new releases.

### ✅ Implemented
- Core `EliminationSearchCV` class with `fit()`, `best_params_`, `best_score_`, `best_estimator_`
- Round 1: per-parameter isolation and elimination
- Rounds 2+: global combination ranking and elimination
- Cross-validated fold creation (`StratifiedKFold` / `KFold`)
- Invalid combination handling (score `0.0`)
- Scoring utilities for 5 metrics

### 🔨 In Progress / Planned
- `cv_results_` attribute (per-fold score breakdown)
- `n_jobs` parallel evaluation via `joblib`
- `verbose` logging parameter
- `refit` flag (opt-out of best-estimator refit)
- Scikit-Learn `BaseEstimator` compatibility (`get_params` / `set_params`)
- Full `pytest` test suite
- PyPI publication (`v0.1.0`)
- Sphinx / MkDocs API documentation

---

## 🤝 Contributing

Contributions of any size are welcome — from fixing a typo in the docs to implementing `n_jobs` parallel fitting.

1. **Read [CONTRIBUTING.md](./CONTRIBUTING.md)** for setup instructions, branch naming, and PR checklist.
2. **Browse open issues** at [github.com/thisal-d/elimination-search-cv/issues](https://github.com/thisal-d/elimination-search-cv/issues)
3. **Open an issue** before starting significant work — aligns design and avoids duplicate effort.
4. **Fork → feature branch → PR** against `main` (e.g. `feat/n-jobs-parallel`).
5. PRs should include tests and clean docstrings.

Not ready to code? You can still help by:
- ⭐ Starring the repo to boost visibility
- Reporting bugs or missing features via [GitHub Issues](https://github.com/thisal-d/elimination-search-cv/issues)
- Sharing benchmarks or datasets where elimination behaves unexpectedly

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