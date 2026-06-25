# Algorithm — How EliminationSearchCV Works (v0.0.1)

This page explains the elimination logic in detail — what happens inside `fit()`, why the two-round strategy exists, and when it helps or hurts.

---

## The Core Idea

`GridSearchCV` evaluates every combination in the Cartesian product of your parameter grid. For a grid with 4 parameters and 6 values each, that's `6⁴ = 1296` combinations × `cv` folds.

`EliminationSearchCV` takes a different path: **it tests parameter values in isolation first, prunes the underperformers, then gradually builds up to full combinations using only the surviving values.**

The intuition is: *if `learning_rate=0.5` consistently scores badly when tested alone, it's unlikely to become competitive when combined with other parameters.* So we eliminate it early rather than carrying it into all downstream combinations.

---

## Round Structure

For a grid with `k` parameters, `fit()` runs exactly `k` rounds:

```
Round 1  →  test each value in isolation (1-parameter combos)
             └─ prune per-parameter: keep top (1 - elimination_rate) values per param

Round 2  →  test all 2-parameter combinations from surviving values
             └─ prune globally: rank all combos, keep top (1 - elimination_rate) fraction

Round 3  →  test all 3-parameter combinations from surviving values
             └─ prune globally

...

Round k  →  test all k-parameter combinations (full combos)
             └─ winner = the best full combination
```

After all rounds, `_active_param_grid` has converged to a single value per parameter. That becomes `best_params_`.

---

## Round 1: Per-Parameter Elimination (Isolation Scoring)

In Round 1, each parameter is evaluated **independently**. A candidate like `{'C': 0.1}` is scored by fitting the estimator with only `C=0.1` set (all other params stay at their estimator defaults).

**Why score in isolation?**
Because comparing `C=0.1` against `C=10` fairly requires that every other variable is held equal. If we scored `{C=0.1, penalty='l1'}` against `{C=10, penalty='l2'}` in round 1, we couldn't tell whether the score difference is due to `C` or `penalty`.

### Worked Example

```
param_grid = {
    'C':       [0.001, 0.01, 0.1, 1, 10, 100],   # 6 values
    'penalty': ['l1', 'l2'],                        # 2 values
    'solver':  ['liblinear', 'saga'],               # 2 values
}
elimination_rate = 0.8   →   keep best 20%
```

**Round 1 scoring:**

| Candidate | CV Score |
|---|---|
| `{C: 0.001}` | 0.71 |
| `{C: 0.01}` | 0.74 |
| `{C: 0.1}` | 0.87 |
| `{C: 1}` | **0.92** |
| `{C: 10}` | 0.91 |
| `{C: 100}` | 0.89 |
| `{penalty: 'l1'}` | **0.92** |
| `{penalty: 'l2'}` | 0.90 |
| `{solver: 'liblinear'}` | **0.92** |
| `{solver: 'saga'}` | 0.90 |

**Elimination per parameter:**

For `C`: 6 values, keep `max(1, round(6 × 0.2)) = 1` value → keep `C=1`
For `penalty`: 2 values, keep `max(1, round(2 × 0.2)) = 1` value → keep `penalty='l1'`
For `solver`: 2 values, keep `max(1, round(2 × 0.2)) = 1` value → keep `solver='liblinear'`

**Active grid after Round 1:**
```python
{'C': [1], 'penalty': ['l1'], 'solver': ['liblinear']}
```

Since every parameter is already down to 1 value, Rounds 2 and 3 each have only 1 combination to score. The result is `best_params_ = {'C': 1, 'penalty': 'l1', 'solver': 'liblinear'}`.

---

## Rounds 2+: Global Combination Ranking

In later rounds, candidates are multi-parameter combinations. Instead of pruning per-parameter, all candidates are **ranked globally by score** and only the top `(1 - elimination_rate)` fraction is kept.

### Worked Example (Round 2 with more surviving values)

Suppose after Round 1 the active grid is:
```python
{'C': [0.1, 1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear']}
```

Round 2 generates all 2-parameter combinations:

| Candidate | CV Score |
|---|---|
| `{C: 0.1, penalty: 'l1'}` | 0.87 |
| `{C: 0.1, penalty: 'l2'}` | 0.86 |
| `{C: 1,   penalty: 'l1'}` | **0.92** |
| `{C: 1,   penalty: 'l2'}` | 0.90 |
| `{C: 10,  penalty: 'l1'}` | 0.91 |
| `{C: 10,  penalty: 'l2'}` | 0.89 |
| `{C: 0.1, solver: 'liblinear'}` | 0.87 |
| ... | ... |

With `elimination_rate=0.8` and 10 candidates, `n_keep = max(1, round(10 × 0.2)) = 2`.

The top 2 combos are kept. Unique values from those 2 combos are extracted per parameter and the active grid is updated. Any parameter that didn't appear in the top 2 combos **keeps its current values unchanged** (not zeroed out).

---

## The "Always Keep ≥ 1 Value" Rule

Across all rounds, this invariant is enforced:

```python
n_keep = max(1, round(n_total * (1 - elimination_rate)))
```

Even if `elimination_rate=0.99`, at least 1 value per parameter survives. This prevents the grid from collapsing to an empty state.

---

## Invalid Combinations

Some estimators raise an exception for certain parameter combinations. For example, `LogisticRegression` does not support `penalty='l1'` with `solver='lbfgs'`.

`EliminationSearchCV` handles this by wrapping every `model.fit()` in a `try/except`:

```python
try:
    model.fit(X_train, y_train)
except Exception:
    score = 0.0   # mark as invalid
```

Invalid combinations receive a score of `0.0`. Since they rank at the bottom, they are naturally eliminated in the next pruning step. **No special handling is needed in your parameter grid** — you can include incompatible values freely.

---

## What `best_score_` Measures

After all rounds complete, the winning `best_params_` is re-scored over the same CV folds that were built at the start of `fit()`. The mean of those fold scores is stored as `best_score_`.

This is consistent with `GridSearchCV.best_score_` — it is the **cross-validated training score**, not a test-set score.

---

## Known Failure Modes

### Small Grids (< ~20 total combinations)

For small search spaces, the overhead of running `k` rounds of scoring outweighs any savings from elimination. On a 4-combination grid, elimination still runs 4 rounds, each scoring the same 1–4 candidates. `GridSearchCV` is faster in this regime.

The benchmarks confirm this: light (small) grid runs show EliminationSearchCV at 0.1–0.6x the speed of GridSearchCV. See [Benchmarks](./benchmarks.md).

### Noisy Cross-Validation (Small Datasets)

If the dataset is small (< 500 rows) or the CV score variance is high, good parameter values may score poorly by chance in Round 1 and get eliminated before they can compete in later rounds.

**Mitigation:** lower `elimination_rate` (e.g. `0.5`–`0.6`) so fewer values are pruned per round.

### Correlated Parameters

If two parameters have strong interaction effects (their combined performance is much better than either in isolation), Round 1 may not correctly rank the individual values. The global ranking in later rounds partially recovers from this, but it is a fundamental limitation of the elimination approach.

---

## Complexity Comparison

| | `GridSearchCV` | `EliminationSearchCV` (best case) |
|---|---|---|
| Combinations evaluated | `∏ len(values_i)` | Shrinks each round |
| CV fits per combination | `cv` | `cv` |
| Total fits | `∏ len(values_i) × cv` | Much smaller for large grids |
| Memory | O(combinations) | O(active grid) |

For the 4-parameter LogisticRegression example from the README (6×2×2×2 = 48 combinations, cv=5):

- `GridSearchCV`: 48 × 5 = **240 fits**
- `EliminationSearchCV` with `elimination_rate=0.8`: **≈ 23 fits**
