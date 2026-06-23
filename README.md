# EliminationSearchCV 🚀

An experimental hyperparameter search library for Scikit-Learn. The core idea is to **prune poorly-performing parameter values early**, before they pollute the rest of the search, rather than blindly evaluating every combination in the full grid. Whether this approach holds up across diverse models and datasets is something this project aims to find out.

---

## 💡 The Core Problem & Our Solution

### The Problem with Standard `GridSearchCV`

Scikit-Learn's `GridSearchCV` is a cornerstone tool, but its underlying strategy is fundamentally **brute-force**. Given a parameter grid with `k` hyperparameters, each with `n` candidate values, it evaluates every combination in the full Cartesian product — that's `n^k` configurations, multiplied by your number of CV folds.

This creates a **combinatorial explosion** that scales poorly and wastes enormous amounts of compute time:

- **Dead-end values are never discarded.** A `learning_rate=0.5` that tanks performance on the very first trial will still be re-evaluated in thousands of later combinations.
- **All configurations are treated as equally promising.** There is no mechanism to learn from early results and steer the search toward more fertile regions.
- **Cost scales exponentially with grid size.** Adding one new hyperparameter with 4 candidate values can quadruple your total training time.

For large models, big datasets, or expensive cross-validation, this is not just slow — it is **prohibitively costly**.

---

### The Elimination Search Solution

`EliminationSearchCV` breaks the search into a series of intelligent, sequential evaluation rounds. Rather than committing to the full Cartesian product upfront, it **observes, learns, and prunes** the search space as it goes.

**Here is a concrete example.** Suppose you are tuning a gradient-boosted model with the following grid:

```
learning_rate : [0.001, 0.01, 0.1, 0.5]
max_depth     : [3, 5, 10]
n_estimators  : [50, 100, 200]
```

A full `GridSearchCV` would evaluate all **4 × 3 × 3 = 36 configurations** (× CV folds).

`EliminationSearchCV` proceeds differently:

1. **Round 1 — Anchor Baseline Selection:** A small set of anchor configurations is selected to establish a performance baseline across all hyperparameter dimensions.

2. **Round 2 — Dimension Isolation & Valuation:** Each hyperparameter is evaluated in isolation. For example, the algorithm tests all four `learning_rate` values while holding `max_depth` and `n_estimators` fixed at their anchor (baseline) values. It discovers that `learning_rate=0.5` is a consistent underperformer — its cross-validated score is substantially below the others in every anchor context.

3. **Round 3 — Dynamic Search Space Pruning:** `learning_rate=0.5` is **eliminated from the search space entirely**. No future configuration will ever include it. The active grid now contains only 3 viable `learning_rate` values.

4. **Round 4 — Focused Sub-grid Execution:** The remaining search is executed on the pruned grid: **3 × 3 × 3 = 27 configurations** — or potentially far fewer if additional values are eliminated during subsequent rounds.

> **The idea:** By eliminating just one `learning_rate` value, we cut 9 configurations (25%) from the search after only a handful of targeted evaluations. In theory, savings compound across dimensions — but how much time is actually saved versus a full grid search, and whether the best result is still found reliably, are open questions that benchmarking will answer.

---

## 🛠️ Architecture & How It Works

`EliminationSearchCV` orchestrates its search through four well-defined execution phases:

- **Phase 1 — Anchor Baseline Selection**
  Before any elimination can occur, the algorithm needs a reference point. A compact set of "anchor" configurations is selected from the parameter grid. These anchors serve as a stable, neutral context for evaluating individual hyperparameter values in isolation. The anchor selection strategy is designed to ensure broad coverage of the search space, not just a single default configuration.

- **Phase 2 — Dimension Isolation & Valuation**
  Each hyperparameter **dimension** (e.g., `learning_rate`, `max_depth`) is evaluated independently against the anchor baselines. For each candidate value within a dimension, the algorithm computes a cross-validated performance score while holding all other hyperparameters fixed at their anchor values. This produces a ranked valuation of every candidate value in the grid, dimension by dimension.

- **Phase 3 — Dynamic Search Space Pruning**
  Candidate values whose performance falls below a configurable elimination threshold are **dropped from the active search space**. This pruning is applied after each dimension is scored, meaning that the search space contracts progressively and dynamically. Eliminated values are never revisited. Each eliminated value reduces the size of the final Cartesian product — and this reduction compounds multiplicatively across all dimensions.

- **Phase 4 — Focused Sub-grid Execution**
  With the search space now pruned to contain only the most promising candidate values, a final, targeted grid search is executed over the remaining sub-grid. Because this sub-grid is significantly smaller than the original, this final phase is fast. The result — `best_params_`, `best_score_`, and `cv_results_` — is surfaced through a fully Scikit-Learn-compatible interface.

---

## 💻 API Reference & Usage Example

> **Note:** The API is still being designed. The three core parameters — `estimator`, `param_grid`, and `scoring` — are confirmed. Everything else (additional parameters, exact attribute names, return types) is not finalized yet and may change as implementation progresses.

### Installation

```bash
pip install elimination-search-cv
```

### Basic Usage

```python
from sklearn.datasets import make_classification
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

from elimination_search_cv import EliminationSearchCV

# --- 1. Prepare Data ---
X, y = make_classification(n_samples=1000, n_features=20, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# --- 2. Define Your Model & Parameter Grid ---
#         (Exactly the same as you would for GridSearchCV)
model = RandomForestClassifier(random_state=42)

param_grid = {
    'n_estimators':     [50, 100, 200, 500],
    'max_depth':        [None, 5, 10, 20],
    'min_samples_split': [2, 5, 10, 20],
    'max_features':     ['sqrt', 'log2', 0.5],
}

# --- 3. Initialize EliminationSearchCV ---
#         Only the three confirmed parameters are shown here.
#         Other parameters (e.g. cv, verbose, n_jobs) are not decided yet.
search = EliminationSearchCV(
    estimator=model,
    param_grid=param_grid,
    scoring='accuracy',
)

# --- 4. Fit ---
search.fit(X_train, y_train)

# --- 5. Access Results ---
#         Exact attribute names are still being figured out.
#         The goal is to expose at least best_params_ and best_score_.
print(search.best_params_)
print(search.best_score_)
```

### Confirmed Parameters

These three parameters are the foundation of the API. Their behaviour is settled:

| Parameter | Description |
|---|---|
| `estimator` | Any Scikit-Learn compatible estimator — something with a `fit` method. |
| `param_grid` | A dictionary mapping parameter names to lists of values to search over. |
| `scoring` | The metric used to evaluate and compare configurations (e.g. `'accuracy'`, `'f1'`). |

### Everything Else — Not Decided Yet

Parameters like `cv`, `verbose`, and `n_jobs` are not confirmed. They are common in `GridSearchCV` and will likely appear in some form, but the exact names, defaults, and behaviour are still being worked out during implementation.

The same applies to post-fit attributes. The intention is to expose at least `best_params_` and `best_score_`, but the full attribute surface is TBD.

---

## 🚧 Project Status: Early Stage, Building in Public

`EliminationSearchCV` is at an early stage — the algorithm is designed, but the implementation has not started yet. This project is being built openly so that progress, decisions, and dead-ends are all visible.

This might work out well, or it might turn out the elimination heuristic is too aggressive in some cases and misses good configurations. That is what testing and real benchmarks are for. If you are interested in the idea and want to follow along, contribute, or challenge the approach — feel free to open an issue or jump into the roadmap below.

### 📋 Roadmap & Todo

#### ✅ Foundation

- [x] Initial project scaffolding (`pyproject.toml`, `src` layout, `LICENSE`)
- [x] Core `README.md` drafted with algorithm description and API reference
- [x] Repository made public and open for contributions

#### 🔨 Core Implementation

- [ ] Implement abstract `BaseEliminationSearch` class with Scikit-Learn `BaseEstimator` compatibility
- [ ] Implement `EliminationSearchCV` class extending the base class
- [ ] Implement **Phase 1**: Anchor baseline configuration selection logic
- [ ] Implement **Phase 2**: Per-dimension hyperparameter isolation and cross-validated scoring
- [ ] Implement **Phase 3**: Elimination threshold logic and dynamic search space pruning
- [ ] Implement **Phase 4**: Final focused sub-grid search over pruned parameter space
- [ ] Expose `eliminated_params_` attribute for post-fit inspection of pruned values
- [ ] Ensure full compatibility with the `cv_results_` dictionary schema from `GridSearchCV`

#### 🧪 Testing & Validation

- [ ] Set up `pytest` test suite in the `tests/` directory
- [ ] Unit tests for anchor selection, elimination logic, and sub-grid construction
- [ ] Integration tests: verify `EliminationSearchCV` is a valid drop-in for `GridSearchCV` on standard datasets
- [ ] Correctness tests: confirm that pruned values are consistently underperformers, not false positives
- [ ] Edge case tests: single-value dimensions, all values eliminated, single CV fold, etc.

#### ⚡ Performance & Parallelism

- [ ] Implement `n_jobs` parameter with `joblib` for parallel cross-validation evaluation
- [ ] Benchmark `EliminationSearchCV` vs. `GridSearchCV` on a suite of standard datasets and model types
- [ ] Profile and optimize the elimination scoring loop for large parameter grids
- [ ] Explore caching strategies to avoid redundant model fits across dimensions

#### 📦 Distribution & Documentation

- [ ] Publish first release (`v0.1.0`) to PyPI
- [ ] Write full API documentation with `sphinx` or `mkdocs-material`
- [ ] Add Jupyter Notebook demo showing real-world speedup comparisons
- [ ] Add GitHub Actions CI pipeline for automated testing on push/PR

---

## 🤝 Contributing & License

### How to Contribute

`EliminationSearchCV` is an open-source project and **contributions of all kinds are enthusiastically welcomed**. You do not need to be a hyperparameter tuning expert to make a meaningful impact — there are tasks at every level of complexity on the roadmap above.

Here is how to get involved:

1. **Browse the Roadmap.** Find a task in the checklist above that interests you. Unchecked items in the *Core Implementation* and *Testing & Validation* sections are the most immediate priorities.

2. **Open an Issue.** Before starting work on a significant change, open a GitHub Issue to describe what you want to do. This lets us align on the design, avoid duplicate work, and give you early feedback. For small fixes, feel free to go straight to a Pull Request.

3. **Fork & Submit a PR.** Fork the repository, create a feature branch (e.g., `feat/phase-2-dimension-scoring`), implement your changes with clean code and docstrings, and open a Pull Request against `main`. PRs should include relevant tests wherever applicable.

4. **Report Bugs & Request Features.** Found unexpected behaviour? Have an idea for a new feature? Open a GitHub Issue with a clear description, a minimal reproducible example if relevant, and any context that would help us investigate.

5. **Improve Documentation.** Clear, accurate documentation is just as valuable as code. If you find anything confusing, incomplete, or out of date — fix it and send a PR.

> **All contributors are expected to engage respectfully and constructively.** This is a welcoming space for developers at every experience level.

### License

This project is licensed under the **MIT License**. See the [LICENSE](./LICENSE) file for the full license text.

You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of this software, subject to the conditions of the MIT License.

---

<p align="center">
  Started by <a href="https://github.com/thisal-d">Thisal-D</a>. A work in progress.
</p>