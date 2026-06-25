# Scaling Benchmark Results (v0.0.1) — EliminationSearchCV vs GridSearchCV

> Compares EliminationSearchCV vs GridSearchCV on both **Light** (small) and **Full** (large) grids.
> **EliminationSearchCV Package Version:** `0.0.1`  
> **Settings:** `cv=2` · `elimination_rate=0.8` · `primary_scoring=accuracy` · `sample_size=10,000`  
> **Bold** = winner for that metric. Reproduce with `python benchmarks/benchmark.py`.

---

### LogisticRegression

| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Cancer Level | Light | ✅ Match | ✅ Equal | **0.0000** | **0.0000** | **0.00s** | 0.00s | **0.0x** |
| Cancer Level | Full | ❌ Diff | ✅ Equal | **1.0000** | **1.0000** | **1.76s** | 7.79s | **4.4x** |
| Diabetes | Light | ❌ Diff | ❌ Lower (-0.0011) | 0.9601 | **0.9612** | **0.25s** | 4.29s | **17.0x** |
| Diabetes | Full | ❌ Diff | ❌ Lower (-0.0011) | 0.9601 | **0.9612** | **1.06s** | 3.92s | **3.7x** |
| Stroke | Light | ✅ Match | ✅ Equal | **0.9513** | **0.9513** | 0.07s | **0.03s** | **0.4x** |
| Stroke | Full | ❌ Diff | ✅ Equal | **0.9513** | **0.9513** | **0.49s** | 1.91s | **3.9x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |
|---|---|---|---|
| Cancer Level | Light | `` | `` |
| Cancer Level | Full | `C=1, class_weight=None, fit_intercept=True, max_iter=100, penalty=l2, solver=saga` | `C=1, class_weight=None, fit_intercept=True, max_iter=100, penalty=l1, solver=saga` |
| Diabetes | Light | `C=0.1, penalty=l2, solver=liblinear` | `C=10.0, penalty=l2, solver=liblinear` |
| Diabetes | Full | `C=0.1, class_weight=None, fit_intercept=True, max_iter=100, penalty=l2, solver=liblinear` | `C=10, class_weight=None, fit_intercept=True, max_iter=100, penalty=l2, solver=liblinear` |
| Stroke | Light | `C=0.1, penalty=l2, solver=liblinear` | `C=0.1, penalty=l2, solver=liblinear` |
| Stroke | Full | `C=0.001, class_weight=None, fit_intercept=True, max_iter=100, penalty=l2, solver=liblinear` | `C=0.001, class_weight=None, fit_intercept=True, max_iter=100, penalty=l1, solver=liblinear` |

</details>

---

### RandomForest

| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Cancer Level | Light | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 0.83s | **0.18s** | **0.2x** |
| Cancer Level | Full | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | **16.21s** | 621.63s | **38.3x** |
| Diabetes | Light | ✅ Match | ✅ Equal | **0.9718** | **0.9718** | 1.59s | **0.67s** | **0.4x** |
| Diabetes | Full | ❌ Diff | ❌ Lower (-0.0003) | 0.9718 | **0.9721** | **65.42s** | 1257.77s | **19.2x** |
| Stroke | Light | ✅ Match | ✅ Equal | **0.9513** | **0.9513** | 1.02s | **0.21s** | **0.2x** |
| Stroke | Full | ❌ Diff | ❌ Lower (-0.0002) | 0.9513 | **0.9515** | **19.12s** | 972.96s | **50.9x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |
|---|---|---|---|
| Cancer Level | Light | `max_depth=None, n_estimators=10` | `max_depth=None, n_estimators=10` |
| Cancer Level | Full | `criterion=gini, max_depth=None, max_features=sqrt, min_samples_leaf=1, min_samples_split=2, n_estimators=10` | `criterion=gini, max_depth=None, max_features=sqrt, min_samples_leaf=1, min_samples_split=2, n_estimators=10` |
| Diabetes | Light | `max_depth=5, n_estimators=50` | `max_depth=5, n_estimators=50` |
| Diabetes | Full | `criterion=gini, max_depth=5, max_features=sqrt, min_samples_leaf=8, min_samples_split=15, n_estimators=300` | `criterion=gini, max_depth=10, max_features=sqrt, min_samples_leaf=2, min_samples_split=10, n_estimators=10` |
| Stroke | Light | `max_depth=5, n_estimators=10` | `max_depth=5, n_estimators=10` |
| Stroke | Full | `criterion=entropy, max_depth=5, max_features=sqrt, min_samples_leaf=2, min_samples_split=10, n_estimators=10` | `criterion=gini, max_depth=10, max_features=sqrt, min_samples_leaf=1, min_samples_split=10, n_estimators=100` |

</details>

---

### DecisionTree

| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Cancer Level | Light | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 0.02s | **0.02s** | **0.7x** |
| Cancer Level | Full | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | **0.31s** | 72.97s | **234.2x** |
| Diabetes | Light | ❌ Diff | ✅ Equal | **0.9718** | **0.9718** | 0.11s | **0.05s** | **0.5x** |
| Diabetes | Full | ❌ Diff | ✅ Equal | **0.9718** | **0.9718** | **1.08s** | 95.48s | **88.6x** |
| Stroke | Light | ❌ Diff | ❌ Lower (-0.0002) | 0.9450 | **0.9452** | 0.06s | **0.03s** | **0.6x** |
| Stroke | Full | ❌ Diff | ❌ Lower (-0.0023) | 0.9491 | **0.9515** | **0.56s** | 75.99s | **134.7x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |
|---|---|---|---|
| Cancer Level | Light | `max_depth=None, min_samples_split=2` | `max_depth=None, min_samples_split=2` |
| Cancer Level | Full | `criterion=gini, max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2, splitter=best` | `criterion=gini, max_depth=None, max_features=None, min_samples_leaf=1, min_samples_split=2, splitter=best` |
| Diabetes | Light | `max_depth=5, min_samples_split=5` | `max_depth=5, min_samples_split=2` |
| Diabetes | Full | `criterion=entropy, max_depth=5, max_features=None, min_samples_leaf=12, min_samples_split=20, splitter=best` | `criterion=gini, max_depth=3, max_features=None, min_samples_leaf=1, min_samples_split=2, splitter=best` |
| Stroke | Light | `max_depth=5, min_samples_split=5` | `max_depth=5, min_samples_split=2` |
| Stroke | Full | `criterion=gini, max_depth=3, max_features=sqrt, min_samples_leaf=12, min_samples_split=20, splitter=best` | `criterion=gini, max_depth=15, max_features=sqrt, min_samples_leaf=4, min_samples_split=15, splitter=random` |

</details>

---

### KNeighbors

| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Cancer Level | Light | ❌ Diff | ✅ Equal | **1.0000** | **1.0000** | 0.04s | **0.03s** | **0.6x** |
| Cancer Level | Full | ❌ Diff | ✅ Equal | **1.0000** | **1.0000** | **0.73s** | 8.80s | **12.1x** |
| Diabetes | Light | ✅ Match | ✅ Equal | **0.9571** | **0.9571** | 1.08s | **0.26s** | **0.2x** |
| Diabetes | Full | ❌ Diff | ❌ Lower (-0.0011) | 0.9578 | **0.9589** | **16.39s** | 215.49s | **13.2x** |
| Stroke | Light | ✅ Match | ✅ Equal | **0.9485** | **0.9485** | 0.57s | **0.11s** | **0.2x** |
| Stroke | Full | ❌ Diff | ✅ Equal | **0.9513** | **0.9513** | **9.18s** | 82.63s | **9.0x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |
|---|---|---|---|
| Cancer Level | Light | `n_neighbors=3, weights=distance` | `n_neighbors=3, weights=uniform` |
| Cancer Level | Full | `algorithm=auto, metric=manhattan, n_neighbors=1, p=1, weights=distance` | `algorithm=auto, metric=euclidean, n_neighbors=1, p=1, weights=uniform` |
| Diabetes | Light | `n_neighbors=5, weights=uniform` | `n_neighbors=5, weights=uniform` |
| Diabetes | Full | `algorithm=auto, metric=euclidean, n_neighbors=7, p=2, weights=uniform` | `algorithm=auto, metric=minkowski, n_neighbors=7, p=3, weights=distance` |
| Stroke | Light | `n_neighbors=5, weights=uniform` | `n_neighbors=5, weights=uniform` |
| Stroke | Full | `algorithm=auto, metric=manhattan, n_neighbors=11, p=1, weights=uniform` | `algorithm=auto, metric=euclidean, n_neighbors=21, p=1, weights=uniform` |

</details>

---

### GradientBoosting

| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Cancer Level | Light | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 2.63s | **0.33s** | **0.1x** |
| Cancer Level | Full | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | **34.51s** | 1195.49s | **34.6x** |
| Diabetes | Light | ✅ Match | ✅ Equal | **0.9718** | **0.9718** | 3.02s | **0.49s** | **0.2x** |
| Diabetes | Full | ❌ Diff | ❌ Lower (-0.0571) | 0.9150 | **0.9721** | **44.42s** | 1713.63s | **38.6x** |
| Stroke | Light | ✅ Match | ✅ Equal | **0.9497** | **0.9497** | 2.28s | **0.34s** | **0.1x** |
| Stroke | Full | ❌ Diff | ❌ Lower (-0.0010) | 0.9507 | **0.9517** | **39.46s** | 1316.85s | **33.4x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |
|---|---|---|---|
| Cancer Level | Light | `learning_rate=0.1, max_depth=3, n_estimators=50` | `learning_rate=0.1, max_depth=3, n_estimators=50` |
| Cancer Level | Full | `learning_rate=0.01, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=50, subsample=0.6` | `learning_rate=0.01, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=50, subsample=0.6` |
| Diabetes | Light | `learning_rate=0.1, max_depth=3, n_estimators=50` | `learning_rate=0.1, max_depth=3, n_estimators=50` |
| Diabetes | Full | `learning_rate=0.01, max_depth=3, min_samples_leaf=4, min_samples_split=5, n_estimators=50, subsample=1.0` | `learning_rate=0.05, max_depth=4, min_samples_leaf=4, min_samples_split=10, n_estimators=150, subsample=1.0` |
| Stroke | Light | `learning_rate=0.1, max_depth=3, n_estimators=50` | `learning_rate=0.1, max_depth=3, n_estimators=50` |
| Stroke | Full | `learning_rate=0.05, max_depth=3, min_samples_leaf=1, min_samples_split=2, n_estimators=100, subsample=1.0` | `learning_rate=0.05, max_depth=3, min_samples_leaf=2, min_samples_split=10, n_estimators=100, subsample=1.0` |

</details>

---

### Speed & Score Summary (Light vs Full Grid Scaling)

| Model | Grid Size | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| DecisionTree | Light | **0.06s** | 0.03s | **0.6x** | -0.0001 |
| DecisionTree | Full | **0.65s** | 81.48s | **152.5x** | -0.0008 |
| GradientBoosting | Light | **2.64s** | 0.39s | **0.1x** | +0.0000 |
| GradientBoosting | Full | **39.46s** | 1408.66s | **35.5x** | -0.0194 |
| KNeighbors | Light | **0.56s** | 0.13s | **0.3x** | +0.0000 |
| KNeighbors | Full | **8.77s** | 102.31s | **11.4x** | -0.0004 |
| LogisticRegression | Light | **0.11s** | 1.44s | **5.8x** | -0.0004 |
| LogisticRegression | Full | **1.10s** | 4.54s | **4.0x** | -0.0004 |
| RandomForest | Light | **1.15s** | 0.35s | **0.3x** | +0.0000 |
| RandomForest | Full | **33.58s** | 950.79s | **36.2x** | -0.0002 |

---

> Environment: Python 3.12, scikit-learn, Windows 11
> Results may vary slightly on re-runs due to random sampling and CV splits.