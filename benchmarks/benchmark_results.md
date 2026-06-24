# Benchmark Results — EliminationSearchCV vs GridSearchCV

> **Settings:** `cv=5` · `elimination_rate=0.8` · `primary_scoring=accuracy` · `sample_size=20,000`  
> **Bold** = winner for that metric. Reproduce with `python benchmarks/benchmark.py`.

---

### LogisticRegression

| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | ❌ Diff | ❌ Lower (-0.0502) | 0.7723 | **0.8224** | **0.25s** | 3.51s | **14.0x** |
| Cancer Level | 1,000 | ❌ Diff | ❌ Lower (-0.0010) | 0.9990 | **1.0000** | **1.33s** | 5.78s | **4.3x** |
| Diabetes | 20,000 | ❌ Diff | ❌ Lower (-0.0005) | 0.9592 | **0.9597** | **2.12s** | 3.63s | **1.7x** |
| Stroke | 5,110 | ❌ Diff | ❌ Lower (-0.0002) | 0.9513 | **0.9515** | **0.59s** | 0.68s | **1.1x** |
| Divorce | 170 | ❌ Diff | ✅ Equal | **0.9765** | **0.9765** | **0.36s** | 1.58s | **4.4x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `C=1, max_iter=500, penalty=l2, solver=saga` | `C=0.01, max_iter=500, penalty=l2, solver=liblinear` |
| Cancer Level | `C=0.1, max_iter=500, penalty=l2, solver=saga` | `C=1, max_iter=500, penalty=l1, solver=saga` |
| Diabetes | `C=10, max_iter=500, penalty=l2, solver=liblinear` | `C=0.1, max_iter=500, penalty=l1, solver=liblinear` |
| Stroke | `C=0.01, max_iter=500, penalty=l2, solver=liblinear` | `C=0.1, max_iter=500, penalty=l2, solver=saga` |
| Divorce | `C=1, max_iter=500, penalty=l2, solver=liblinear` | `C=0.01, max_iter=500, penalty=l2, solver=liblinear` |

#### 📊 Secondary Metrics (F1, Precision, Recall)

| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.6975 | **0.7762** | 0.7185 | **0.8503** | 0.7231 | **0.7935** |
| Cancer Level | 0.9989 | **1.0000** | 0.9990 | **1.0000** | 0.9989 | **1.0000** |
| Diabetes | 0.8498 | **0.8498** | 0.9122 | **0.9195** | **0.8070** | 0.8035 |
| Stroke | 0.4875 | **0.4915** | 0.4756 | **0.5757** | 0.5000 | **0.5020** |
| Divorce | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** |

</details>

---

### RandomForest

| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | ❌ Diff | ❌ Lower (-0.0301) | 0.7123 | **0.7424** | 10.84s | **8.86s** | **0.8x** |
| Cancer Level | 1,000 | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 11.47s | **10.74s** | **0.9x** |
| Diabetes | 20,000 | ❌ Diff | ✅ Equal | **0.9712** | **0.9712** | **20.89s** | 33.03s | **1.6x** |
| Stroke | 5,110 | ❌ Diff | ✅ Equal | **0.9513** | **0.9513** | 16.99s | **16.12s** | **0.9x** |
| Divorce | 170 | ✅ Match | ✅ Equal | **0.9765** | **0.9765** | **9.57s** | 10.37s | **1.1x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `max_depth=None, min_samples_split=5, n_estimators=200` | `max_depth=5, min_samples_split=10, n_estimators=50` |
| Cancer Level | `max_depth=None, min_samples_split=2, n_estimators=50` | `max_depth=None, min_samples_split=2, n_estimators=50` |
| Diabetes | `max_depth=5, min_samples_split=10, n_estimators=50` | `max_depth=5, min_samples_split=2, n_estimators=50` |
| Stroke | `max_depth=5, min_samples_split=10, n_estimators=200` | `max_depth=5, min_samples_split=2, n_estimators=50` |
| Divorce | `max_depth=None, min_samples_split=2, n_estimators=50` | `max_depth=None, min_samples_split=2, n_estimators=50` |

#### 📊 Secondary Metrics (F1, Precision, Recall)

| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.6142 | **0.6621** | 0.6883 | **0.8079** | 0.6724 | **0.7030** |
| Cancer Level | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Diabetes | **0.8902** | **0.8902** | **0.9847** | **0.9847** | **0.8306** | **0.8306** |
| Stroke | **0.4875** | **0.4875** | **0.4756** | **0.4756** | **0.5000** | **0.5000** |
| Divorce | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** |

</details>

---

### DecisionTree

| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | ❌ Diff | ❌ Lower (-0.0067) | 0.6589 | **0.6656** | 0.12s | **0.11s** | **0.9x** |
| Cancer Level | 1,000 | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 0.16s | **0.13s** | **0.8x** |
| Diabetes | 20,000 | ❌ Diff | ✅ Equal | **0.9712** | **0.9712** | 1.42s | **1.19s** | **0.8x** |
| Stroke | 5,110 | ❌ Diff | ❌ Lower (-0.0008) | 0.9476 | **0.9483** | 0.53s | **0.36s** | **0.7x** |
| Divorce | 170 | ✅ Match | ✅ Equal | **0.9765** | **0.9765** | **0.08s** | 0.11s | **1.3x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `criterion=entropy, max_depth=5, min_samples_split=10` | `criterion=entropy, max_depth=5, min_samples_split=5` |
| Cancer Level | `criterion=gini, max_depth=None, min_samples_split=2` | `criterion=gini, max_depth=None, min_samples_split=2` |
| Diabetes | `criterion=entropy, max_depth=5, min_samples_split=10` | `criterion=gini, max_depth=5, min_samples_split=2` |
| Stroke | `criterion=entropy, max_depth=5, min_samples_split=10` | `criterion=gini, max_depth=5, min_samples_split=2` |
| Divorce | `criterion=gini, max_depth=None, min_samples_split=2` | `criterion=gini, max_depth=None, min_samples_split=2` |

#### 📊 Secondary Metrics (F1, Precision, Recall)

| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.5414 | **0.5547** | 0.5892 | **0.6075** | 0.6051 | **0.6156** |
| Cancer Level | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Diabetes | **0.8902** | **0.8902** | **0.9847** | **0.9847** | **0.8306** | **0.8306** |
| Stroke | 0.4931 | **0.5007** | 0.4939 | **0.5426** | 0.5018 | **0.5060** |
| Divorce | **0.9763** | **0.9763** | **0.9794** | **0.9794** | **0.9765** | **0.9765** |

</details>

---

### KNeighbors

| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | ❌ Diff | ❌ Lower (-0.0134) | 0.7256 | **0.7390** | 0.14s | **0.09s** | **0.6x** |
| Cancer Level | 1,000 | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 1.69s | **0.22s** | **0.1x** |
| Diabetes | 20,000 | ❌ Diff | ❌ Lower (-0.0008) | 0.9578 | **0.9586** | 13.12s | **8.57s** | **0.7x** |
| Stroke | 5,110 | ❌ Diff | ❌ Lower (-0.0006) | 0.9507 | **0.9513** | 2.37s | **1.15s** | **0.5x** |
| Divorce | 170 | ✅ Match | ✅ Equal | **0.9765** | **0.9765** | 0.16s | **0.07s** | **0.4x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `metric=manhattan, n_neighbors=5, weights=uniform` | `metric=euclidean, n_neighbors=5, weights=uniform` |
| Cancer Level | `metric=euclidean, n_neighbors=3, weights=uniform` | `metric=euclidean, n_neighbors=3, weights=uniform` |
| Diabetes | `metric=euclidean, n_neighbors=9, weights=uniform` | `metric=euclidean, n_neighbors=15, weights=distance` |
| Stroke | `metric=manhattan, n_neighbors=9, weights=uniform` | `metric=euclidean, n_neighbors=15, weights=uniform` |
| Divorce | `metric=euclidean, n_neighbors=3, weights=uniform` | `metric=euclidean, n_neighbors=3, weights=uniform` |

#### 📊 Secondary Metrics (F1, Precision, Recall)

| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.5961 | **0.6335** | 0.6641 | **0.7776** | 0.6228 | **0.6440** |
| Cancer Level | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Diabetes | 0.8366 | **0.8370** | 0.9298 | **0.9418** | **0.7814** | 0.7775 |
| Stroke | 0.4874 | **0.4875** | 0.4756 | **0.4756** | 0.4997 | **0.5000** |
| Divorce | **0.9761** | **0.9761** | **0.9810** | **0.9810** | **0.9765** | **0.9765** |

</details>

---

### GradientBoosting

| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |
|---|---|---|---|---|---|---|---|---|
| Heart Failure | 299 | ❌ Diff | ❌ Lower (-0.0301) | 0.6489 | **0.6789** | 5.45s | **4.45s** | **0.8x** |
| Cancer Level | 1,000 | ✅ Match | ✅ Equal | **1.0000** | **1.0000** | 22.45s | **16.67s** | **0.7x** |
| Diabetes | 20,000 | ✅ Match | ✅ Equal | **0.9712** | **0.9712** | 62.72s | **45.84s** | **0.7x** |
| Stroke | 5,110 | ✅ Match | ✅ Equal | **0.9507** | **0.9507** | 20.10s | **17.16s** | **0.9x** |
| Divorce | 170 | ❌ Diff | ❌ Lower (-0.0059) | 0.9765 | **0.9824** | 3.41s | **2.37s** | **0.7x** |

<details>
<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>
<br>

#### ⚙️ Selected Best Parameters

| Dataset | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Heart Failure | `learning_rate=0.05, max_depth=5, n_estimators=50` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Cancer Level | `learning_rate=0.05, max_depth=3, n_estimators=50` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Diabetes | `learning_rate=0.05, max_depth=3, n_estimators=50` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Stroke | `learning_rate=0.05, max_depth=3, n_estimators=50` | `learning_rate=0.05, max_depth=3, n_estimators=50` |
| Divorce | `learning_rate=0.05, max_depth=3, n_estimators=50` | `learning_rate=0.1, max_depth=5, n_estimators=200` |

#### 📊 Secondary Metrics (F1, Precision, Recall)

| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |
|---|---|---|---|---|---|---|
| Heart Failure | 0.5359 | **0.5640** | 0.5917 | **0.6119** | 0.6062 | **0.6283** |
| Cancer Level | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** | **1.0000** |
| Diabetes | **0.8902** | **0.8902** | **0.9847** | **0.9847** | **0.8306** | **0.8306** |
| Stroke | **0.4874** | **0.4874** | **0.4756** | **0.4756** | **0.4997** | **0.4997** |
| Divorce | 0.9763 | **0.9822** | 0.9791 | **0.9850** | 0.9768 | **0.9824** |

</details>

---

### Speed & Score Summary

| Model | Grid Combos | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| LogisticRegression | 40 | **0.93s** | 3.04s | **5.1x** | -0.0104 |
| RandomForest | 27 | **13.95s** | 15.82s | **1.1x** | -0.0060 |
| DecisionTree | 24 | **0.46s** | 0.38s | **0.9x** | -0.0015 |
| KNeighbors | 16 | **3.49s** | 2.02s | **0.5x** | -0.0029 |
| GradientBoosting | 18 | **22.82s** | 17.30s | **0.8x** | -0.0072 |

---

> Environment: Python 3.12, scikit-learn, Windows 11
> Results may vary slightly on re-runs due to random sampling and CV splits.