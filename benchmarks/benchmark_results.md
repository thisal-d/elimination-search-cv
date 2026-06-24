# Benchmark Results — EliminationSearchCV vs GridSearchCV

> **Settings:** `cv=5` · `reduce_rate=0.8` · `primary_scoring=accuracy` · `sample_size=5,000`  
> **Bold** = winner for that metric. Reproduce with `python benchmarks/benchmark.py`.

---

### LogisticRegression

| Dataset | n | Acc (Elim) | Acc (Grid) | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | 0.7723 | **0.8224** | 0.6975 | **0.7762** | 0.7185 | **0.8503** | 0.7231 | **0.7935** | **0.25s** | 3.37s | **13.3x** |
| Cancer Level | 1,000 | 0.9990 | **1.0000** | 0.9989 | **1.0000** | 0.9990 | **1.0000** | 0.9989 | **1.0000** | **1.28s** | 4.93s | **3.9x** |
| Diabetes | 5,000 | 0.9596 | **0.9606** | **0.8506** | 0.8499 | 0.9163 | **0.9346** | **0.8061** | 0.7970 | **0.49s** | 0.48s | **1.0x** |
| Stroke | 5,000 | **0.9512** | **0.9512** | **0.4875** | **0.4875** | **0.4756** | **0.4756** | **0.5000** | **0.5000** | **0.54s** | 0.51s | **0.9x** |
| Divorce | 170 | **0.9765** | **0.9765** | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** | **0.40s** | 1.47s | **3.6x** |

---

### RandomForest

| Dataset | n | Acc (Elim) | Acc (Grid) | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | 0.7123 | **0.7424** | 0.6142 | **0.6621** | 0.6883 | **0.8079** | 0.6724 | **0.7030** | **10.50s** | 6.00s | **0.6x** |
| Cancer Level | 1,000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **9.28s** | 9.65s | **1.0x** |
| Diabetes | 5,000 | **0.9720** | **0.9720** | 0.8937 | **0.8943** | **0.9852** | 0.9821 | 0.8353 | **0.8374** | **12.01s** | 12.84s | **1.1x** |
| Stroke | 5,000 | **0.9512** | **0.9512** | **0.4875** | **0.4875** | **0.4756** | **0.4756** | **0.5000** | **0.5000** | **17.32s** | 20.30s | **1.2x** |
| Divorce | 170 | **0.9765** | **0.9765** | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** | **10.39s** | 8.41s | **0.8x** |

---

### DecisionTree

| Dataset | n | Acc (Elim) | Acc (Grid) | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | 0.6589 | **0.6656** | 0.5414 | **0.5547** | 0.5892 | **0.6075** | 0.6051 | **0.6156** | **0.10s** | 0.10s | **1.0x** |
| Cancer Level | 1,000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **0.19s** | 0.14s | **0.7x** |
| Diabetes | 5,000 | **0.9720** | **0.9720** | **0.8937** | **0.8937** | **0.9852** | **0.9852** | **0.8353** | **0.8353** | **0.40s** | 0.24s | **0.6x** |
| Stroke | 5,000 | 0.9464 | **0.9484** | **0.4967** | 0.4942 | 0.5189 | **0.5233** | **0.5033** | 0.5024 | **0.80s** | 0.88s | **1.1x** |
| Divorce | 170 | **0.9765** | **0.9765** | **0.9763** | **0.9763** | **0.9794** | **0.9794** | **0.9765** | **0.9765** | **0.09s** | 0.12s | **1.4x** |

---

### KNeighbors

| Dataset | n | Acc (Elim) | Acc (Grid) | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | 0.7256 | **0.7390** | 0.5961 | **0.6335** | 0.6641 | **0.7776** | 0.6228 | **0.6440** | **0.15s** | 0.07s | **0.5x** |
| Cancer Level | 1,000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.86s** | 0.20s | **0.1x** |
| Diabetes | 5,000 | 0.9540 | **0.9578** | 0.8098 | **0.8358** | **0.9469** | 0.9331 | 0.7444 | **0.7795** | **1.77s** | 0.88s | **0.5x** |
| Stroke | 5,000 | 0.9512 | **0.9514** | 0.4875 | **0.4915** | 0.4756 | **0.5757** | 0.5000 | **0.5020** | **2.92s** | 1.36s | **0.5x** |
| Divorce | 170 | **0.9765** | **0.9765** | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** | **0.20s** | 0.08s | **0.4x** |

---

### GradientBoosting

| Dataset | n | Acc (Elim) | Acc (Grid) | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | 0.6489 | **0.6789** | 0.5359 | **0.5640** | 0.5917 | **0.6119** | 0.6062 | **0.6283** | **4.82s** | 2.92s | **0.6x** |
| Cancer Level | 1,000 | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **26.83s** | 12.01s | **0.4x** |
| Diabetes | 5,000 | **0.9720** | **0.9720** | **0.8937** | **0.8937** | **0.9852** | **0.9852** | **0.8353** | **0.8353** | **18.06s** | 12.00s | **0.7x** |
| Stroke | 5,000 | **0.9514** | **0.9514** | **0.4955** | **0.4955** | **0.6258** | **0.6258** | **0.5040** | **0.5040** | **21.41s** | 16.78s | **0.8x** |
| Divorce | 170 | 0.9765 | **0.9824** | 0.9763 | **0.9822** | 0.9791 | **0.9850** | 0.9768 | **0.9824** | **3.53s** | 2.25s | **0.6x** |

---

### Speed & Score Summary

| Model | Grid Combos | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| LogisticRegression | 40 | **0.59s** | 2.15s | **4.5x** | -0.0104 |
| RandomForest | 27 | **11.90s** | 11.44s | **0.9x** | -0.0060 |
| DecisionTree | 24 | **0.31s** | 0.29s | **1.0x** | -0.0017 |
| KNeighbors | 16 | **1.38s** | 0.52s | **0.4x** | -0.0035 |
| GradientBoosting | 18 | **14.93s** | 9.19s | **0.6x** | -0.0072 |

---

<details>
<summary><strong>Best Params Found (click to expand)</strong></summary>

#### LogisticRegression

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `C=1, penalty=l2, solver=saga, max_iter=500` | `C=0.01, max_iter=500, penalty=l2, solver=liblinear` |
| Cancer Level | `C=0.1, penalty=l2, solver=saga, max_iter=500` | `C=1, max_iter=500, penalty=l1, solver=saga` |
| Diabetes | `C=1, penalty=l2, solver=liblinear, max_iter=500` | `C=0.1, max_iter=500, penalty=l1, solver=liblinear` |
| Stroke | `C=0.01, penalty=l2, solver=liblinear, max_iter=500` | `C=0.01, max_iter=500, penalty=l1, solver=liblinear` |
| Divorce | `C=1, penalty=l2, solver=liblinear, max_iter=500` | `C=0.01, max_iter=500, penalty=l2, solver=liblinear` |

#### RandomForest

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `n_estimators=200, max_depth=None, min_samples_split=5` | `max_depth=5, min_samples_split=10, n_estimators=50` |
| Cancer Level | `n_estimators=50, max_depth=None, min_samples_split=2` | `max_depth=None, min_samples_split=2, n_estimators=50` |
| Diabetes | `n_estimators=50, max_depth=5, min_samples_split=5` | `max_depth=None, min_samples_split=5, n_estimators=100` |
| Stroke | `n_estimators=100, max_depth=5, min_samples_split=10` | `max_depth=5, min_samples_split=2, n_estimators=50` |
| Divorce | `n_estimators=50, max_depth=None, min_samples_split=2` | `max_depth=None, min_samples_split=2, n_estimators=50` |

#### DecisionTree

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `max_depth=5, min_samples_split=10, criterion=entropy` | `criterion=entropy, max_depth=5, min_samples_split=5` |
| Cancer Level | `max_depth=None, min_samples_split=2, criterion=gini` | `criterion=gini, max_depth=None, min_samples_split=2` |
| Diabetes | `max_depth=5, min_samples_split=10, criterion=entropy` | `criterion=entropy, max_depth=5, min_samples_split=2` |
| Stroke | `max_depth=5, min_samples_split=10, criterion=gini` | `criterion=entropy, max_depth=5, min_samples_split=2` |
| Divorce | `max_depth=None, min_samples_split=2, criterion=gini` | `criterion=gini, max_depth=None, min_samples_split=2` |

#### KNeighbors

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `n_neighbors=5, weights=uniform, metric=manhattan` | `metric=euclidean, n_neighbors=5, weights=uniform` |
| Cancer Level | `n_neighbors=3, weights=uniform, metric=euclidean` | `metric=euclidean, n_neighbors=3, weights=uniform` |
| Diabetes | `n_neighbors=9, weights=uniform, metric=euclidean` | `metric=euclidean, n_neighbors=5, weights=uniform` |
| Stroke | `n_neighbors=15, weights=uniform, metric=euclidean` | `metric=euclidean, n_neighbors=9, weights=uniform` |
| Divorce | `n_neighbors=3, weights=uniform, metric=euclidean` | `metric=euclidean, n_neighbors=3, weights=uniform` |

#### GradientBoosting

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `n_estimators=50, learning_rate=0.05, max_depth=5` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Cancer Level | `n_estimators=50, learning_rate=0.05, max_depth=3` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Diabetes | `n_estimators=100, learning_rate=0.05, max_depth=3` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Stroke | `n_estimators=50, learning_rate=0.05, max_depth=3` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Divorce | `n_estimators=50, learning_rate=0.05, max_depth=3` | `learning_rate=0.1, max_depth=5, n_estimators=200` |

</details>

> Environment: Python 3.12, scikit-learn, Windows 11
> Results may vary slightly on re-runs due to random sampling and CV splits.