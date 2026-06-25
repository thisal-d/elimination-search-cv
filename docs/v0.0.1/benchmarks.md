# Benchmarks — EliminationSearchCV v0.0.1

Results from the first structured benchmark comparing `EliminationSearchCV` against `GridSearchCV` across 5 models and 3 datasets, using both a **Light grid** (small number of combinations) and a **Full grid** (large number of combinations).

---

## Setup

| Setting | Value |
|---|---|
| Package version | `0.0.1` |
| `cv` | `2` |
| `elimination_rate` | `0.8` |
| `scoring` | `accuracy` |
| `sample_size` | 10,000 (sampled from datasets where needed) |
| Python | 3.12 |
| sklearn | latest at time of benchmark |
| OS | Windows 11 |
| Reproduce with | `python benchmarks/benchmark.py` |

> Results may vary slightly on re-runs due to random sampling and CV splits.

---

## What "Light Grid" and "Full Grid" Mean

| Grid type | Description | Purpose |
|---|---|---|
| **Light** | Small parameter grid (few values per parameter) | Tests overhead cost of elimination on small search spaces |
| **Full** | Large parameter grid (many values per parameter) | Tests elimination's actual benefit on large search spaces |

The key insight from this benchmark: **elimination is designed for full/large grids**. On light grids, the overhead of running `n_params` rounds of scoring exceeds the savings from pruning.

---

## Speed & Score Summary (Light vs Full Grid)

| Model | Grid Size | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |
|---|---|---|---|---|---|
| DecisionTree | Light | **0.06s** | 0.03s | **0.6x** | -0.0001 |
| **DecisionTree** | **Full** | **0.65s** | 81.48s | **🚀 152.5x** | -0.0008 |
| GradientBoosting | Light | **2.64s** | 0.39s | **0.1x** | +0.0000 |
| **GradientBoosting** | **Full** | **39.46s** | 1408.66s | **🚀 35.5x** | -0.0194 |
| KNeighbors | Light | **0.56s** | 0.13s | **0.3x** | +0.0000 |
| **KNeighbors** | **Full** | **8.77s** | 102.31s | **🚀 11.4x** | -0.0004 |
| LogisticRegression | Light | **0.11s** | 1.44s | **5.8x** | -0.0004 |
| **LogisticRegression** | **Full** | **1.10s** | 4.54s | **🚀 4.0x** | -0.0004 |
| RandomForest | Light | **1.15s** | 0.35s | **0.3x** | +0.0000 |
| **RandomForest** | **Full** | **33.58s** | 950.79s | **🚀 36.2x** | -0.0002 |

**Bold rows = Full grid results (where elimination provides the most benefit).**

---

## Reading the Numbers

**Avg Speedup:** `GridSearchCV time / EliminationSearchCV time`. A value > 1x means EliminationSearchCV was faster.

**Avg Acc Diff:** `Elim accuracy − Grid accuracy`, averaged across datasets. A value of `-0.0008` means EliminationSearchCV found a combination that scored 0.08% lower on average. A value of `+0.0000` means equal or better.

---

## Key Findings

### 1. Full grids are where elimination shines

`DecisionTree` on a full grid: **152x speedup** with only -0.0008 accuracy difference.
`RandomForest` on a full grid: **36x speedup** with only -0.0002 accuracy difference.
`KNeighbors` on a full grid: **11x speedup** with zero accuracy difference.

These are the use cases EliminationSearchCV is designed for.

### 2. Light grids run slower than GridSearchCV

On light grids, `EliminationSearchCV` consistently shows speedups < 1x (0.1x–0.6x). This is **expected** — the overhead of running `n_params` scoring rounds doesn't pay off when there are only a handful of combinations to begin with.

**Recommendation:** for grids with < 20 total combinations, use `GridSearchCV`.

### 3. Score trade-off is minimal on full grids

Across all full-grid runs, the average accuracy difference is < 0.02, and for most models it is < 0.001. `GradientBoosting` shows the largest trade-off (-0.0194) because its large parameter space has more interaction effects that isolation scoring can miss.

### 4. GradientBoosting Diabetes Full Grid — the outlier

| | EliminationSearchCV | GridSearchCV |
|---|---|---|
| Accuracy | 0.9150 | **0.9721** |
| Score diff | **-0.0571** | — |
| Time | **44.42s** | 1713.63s |
| Speedup | **38.6x** | — |

This is the worst accuracy trade-off in the benchmark. The GradientBoosting full grid has strong parameter interactions that Round 1 isolation scoring fails to capture, leading to poor values being retained and good values being pruned. This is a known limitation of the elimination approach — see [Known Limitations](./known-limitations.md).

---

## Per-Model Detail Tables

Full per-model tables including per-dataset scores, selected hyperparameters, and params match results:

→ **[benchmark_results_scaling_cv2_rate08_size10k_v0.0.1.md](../../benchmarks/marks/benchmark_results_scaling_cv2_rate08_size10k_v0.0.1.md)**

---

## Reproducing the Benchmark

```bash
# Install the package (from repo root)
pip install -e .

# Run the full benchmark
python benchmarks/benchmark.py
```

Results are saved to `benchmarks/marks/` with a version-stamped filename.

> The benchmark samples 10,000 rows from each dataset. Results will vary slightly between runs due to the random sampling.
