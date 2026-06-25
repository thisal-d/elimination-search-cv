"""
benchmark.py — Primary EliminationSearchCV vs GridSearchCV Benchmark
======================================================================
Runs a rigorous head-to-head comparison between EliminationSearchCV and
GridSearchCV. Specifically compares how both algorithms scale from small
(Light) to large (Full) parameter grids on the two largest datasets (Diabetes, Stroke).

Usage:
    python benchmarks/benchmark.py
"""

import os
import sys
import time
import datetime
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

# Suppress all python warnings globally and inside child processes
os.environ["PYTHONWARNINGS"] = "ignore"
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_validate, train_test_split
from sklearn.metrics import accuracy_score

# Path setup
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from EliminationSearchCV import EliminationSearchCV

# ---------------------------------------------------------------------------
# Dataset definitions (Two largest datasets)
# ---------------------------------------------------------------------------
DATASET_DIR = REPO_ROOT / "benchmarks" / "datasets"

DATASETS = {
    "Diabetes": {
        "path": DATASET_DIR / "diabetes_prediction_dataset.csv",
        "target": "diabetes",
        "drop": [],
        "sep": ",",
        "categorical": ["gender", "smoking_history"],
    },
    "Stroke": {
        "path": DATASET_DIR / "healthcare-dataset-stroke-data.csv",
        "target": "stroke",
        "drop": ["id"],
        "sep": ",",
        "categorical": ["gender", "ever_married", "work_type",
                        "Residence_type", "smoking_status"],
    },
    "Cancer Level": {
        "path": DATASET_DIR / "cancer patient data sets.csv",
        "target": "Level",
        "drop": ["index", "Patient Id"],
        "sep": ",",
        "categorical": [],
    },
}

# ---------------------------------------------------------------------------
# Hyperparameter Grids (Full vs Light)
# ---------------------------------------------------------------------------

FULL_MODELS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
            "C":             [0.001, 0.01, 0.1, 1, 10, 100, 1000],
            "penalty":       ["l1", "l2"],
            "solver":        ["liblinear", "saga"],
            "fit_intercept": [True, False],
            "class_weight":  [None, "balanced"],
            "max_iter":      [100, 500, 1000],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators":      [10, 50, 100, 200, 300],
            "max_depth":         [None, 5, 10, 15, 20],
            "min_samples_split": [2, 5, 10, 15],
            "min_samples_leaf":  [1, 2, 4, 8],
            "max_features":      ["sqrt", "log2", None],
            "criterion":         ["gini", "entropy", "log_loss"],
        },
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth":         [None, 3, 5, 10, 15, 20, 30],
            "min_samples_split": [2, 5, 10, 15, 20],
            "min_samples_leaf":  [1, 2, 4, 8, 12],
            "criterion":         ["gini", "entropy", "log_loss"],
            "max_features":      [None, "sqrt", "log2"],
            "splitter":          ["best", "random"],
        },
    },
    "KNeighbors": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21, 31],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan", "minkowski", "chebyshev"],
            "algorithm":   ["auto", "ball_tree", "kd_tree", "brute"],
            "p":           [1, 2, 3],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators":      [50, 100, 150, 200, 300],
            "learning_rate":     [0.01, 0.05, 0.1, 0.15, 0.2],
            "max_depth":         [3, 4, 5, 6, 8],
            "min_samples_split": [2, 5, 10],
            "min_samples_leaf":  [1, 2, 4],
            "subsample":         [0.6, 0.8, 1.0],
        },
    },
}

LIGHT_MODELS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
            "C":             [0.1, 1.0, 10.0],
            "penalty":       ["l2"],
            "solver":        ["liblinear"],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators":      [10, 50],
            "max_depth":         [None, 5],
        },
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth":         [None, 5],
            "min_samples_split": [2, 5],
        },
    },
    "KNeighbors": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5],
            "weights":     ["uniform", "distance"],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators":      [50],
            "learning_rate":     [0.1],
            "max_depth":         [3],
        },
    },
}

# ---------------------------------------------------------------------------
# Benchmark configurations (Adjust settings here)
# ---------------------------------------------------------------------------
BENCHMARK_VERSION = "0.0.1"           # Version of EliminationSearchCV
CV_FOLDS = 2                          # CV folds (1 for predefined single train/val split)
ELIMINATION_RATE = 0.8               # Elimination rate
PRIMARY_SCORING = "accuracy"          # Primary scoring metric
EVAL_METRICS = ["accuracy"]           # Target only one scoring metric for speed
SAMPLE_SIZE = 10000                    # Sampling threshold for big datasets

# ---------------------------------------------------------------------------
# Data loading & preprocessing
# ---------------------------------------------------------------------------

def load_and_preprocess(cfg: Dict) -> Tuple[np.ndarray, np.ndarray, str]:
    """Load dataset, encode categoricals, impute nulls, StandardScale."""
    path = cfg["path"]
    if not path.exists():
        return None, None, "FILE NOT FOUND"

    df = pd.read_csv(path, sep=cfg["sep"])
    notes = []

    df = df.drop(columns=[c for c in cfg["drop"] if c in df.columns])

    target_col = cfg["target"]
    y_raw = df[target_col].copy()
    X_df  = df.drop(columns=[target_col])

    # Encode string target
    if y_raw.dtype == object:
        le = LabelEncoder()
        y = le.fit_transform(y_raw)
        notes.append("target label-encoded")
    else:
        y = y_raw.values

    # Encode categorical features
    for col in cfg["categorical"]:
        if col in X_df.columns and X_df[col].dtype == object:
            le = LabelEncoder()
            X_df[col] = le.fit_transform(X_df[col].astype(str))

    # Encode any remaining object columns
    for col in X_df.select_dtypes(include="object").columns:
        le = LabelEncoder()
        X_df[col] = le.fit_transform(X_df[col].astype(str))
        notes.append(f"auto-encoded '{col}'")

    # Impute nulls
    null_count = X_df.isnull().sum().sum()
    if null_count > 0:
        X_df = X_df.fillna(X_df.median(numeric_only=True))
        notes.append(f"imputed {null_count} nulls")

    X = X_df.select_dtypes(include=[np.number]).values

    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    return X, y, "; ".join(notes) if notes else "no transforms"

# ---------------------------------------------------------------------------
# Scoring helper
# ---------------------------------------------------------------------------

def evaluate_params(estimator, best_params: Dict, X, y) -> Dict[str, float]:
    """Evaluate best_params across EVAL_METRICS using a single pass of cross_validate."""
    model = type(estimator)(**{**estimator.get_params(), **best_params})
    scores = cross_validate(model, X, y, cv=CV_FOLDS, scoring=EVAL_METRICS, error_score=0.0)
    results = {}
    for metric in EVAL_METRICS:
        val = scores[f"test_{metric}"]
        results[metric] = float(val.mean()) if len(val) > 0 else 0.0
    return results

# ---------------------------------------------------------------------------
# Search runners
# ---------------------------------------------------------------------------

def run_elimination(estimator, param_grid, X, y):
    search = EliminationSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=CV_FOLDS,
        elimination_rate=ELIMINATION_RATE,
    )
    t0 = time.perf_counter()
    search.fit(X, y)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_params(estimator, search.best_params_, X, y)
    return search.best_params_, metrics, elapsed


def run_grid_search(estimator, param_grid, X, y):
    if CV_FOLDS == 1:
        indices = np.arange(len(X))
        try:
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42, stratify=y
            )
        except Exception:
            train_idx, val_idx = train_test_split(
                indices, test_size=0.2, random_state=42
            )
        cv_split = [(train_idx, val_idx)]
    else:
        cv_split = CV_FOLDS

    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=cv_split,
        error_score=0.0,
        n_jobs=-1,
    )
    t0 = time.perf_counter()
    gs.fit(X, y)
    elapsed = time.perf_counter() - t0
    metrics = evaluate_params(estimator, gs.best_params_, X, y)
    return gs.best_params_, metrics, elapsed


def grid_total_fits(param_grid: Dict) -> int:
    n = 1
    for v in param_grid.values():
        n *= len(v)
    return n * CV_FOLDS

# ---------------------------------------------------------------------------
# Markdown report builder
# ---------------------------------------------------------------------------

def build_model_table(model_name: str, records: List[Dict]) -> str:
    """Creates a table comparing results side-by-side for Light and Full grids."""
    lines = []
    lines.append(f"### {model_name}")
    lines.append("")

    lines.append(
        "| Dataset | Grid Size | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    # Order records logically (Dataset name, then Light first, then Full)
    sorted_recs = sorted(records, key=lambda x: (x["dataset"], x["grid_type"] == "Full"))

    for r in sorted_recs:
        e = r["elim"]
        g = r["grid"]

        def fmt(val):
            return f"{val:.4f}"

        def winner(ev, gv):
            if ev >= gv:
                return f"**{fmt(ev)}**"
            return fmt(ev)

        params_matched = dict(sorted(e["params"].items())) == dict(sorted(g["params"].items()))
        params_match_str = "✅ Match" if params_matched else "❌ Diff"

        elim_score = e['metrics']['accuracy']
        grid_score = g['metrics']['accuracy']
        diff = elim_score - grid_score
        if abs(diff) < 1e-9:
            score_match_str = "✅ Equal"
        elif diff < 0:
            score_match_str = f"❌ Lower ({diff:.4f})"
        else:
            score_match_str = f"🔥 Higher (+{diff:.4f})"

        speedup = g["time"] / e["time"] if e["time"] > 0 else 0

        if e["time"] <= g["time"]:
            time_elim_str = f"**{e['time']:.2f}s**"
            time_grid_str = f"{g['time']:.2f}s"
        else:
            time_elim_str = f"{e['time']:.2f}s"
            time_grid_str = f"**{g['time']:.2f}s**"

        lines.append(
            f"| {r['dataset']} | {r['grid_type']} | "
            f"{params_match_str} | "
            f"{score_match_str} | "
            f"{winner(elim_score, grid_score)} | "
            f"{winner(grid_score, elim_score)} | "
            f"{time_elim_str} | "
            f"{time_grid_str} | "
            f"**{speedup:.1f}x** |"
        )

    lines.append("")
    lines.append("<details>")
    lines.append("<summary><strong>🔍 View Selected Hyperparameters</strong></summary>")
    lines.append("<br>")
    lines.append("")
    lines.append("#### ⚙️ Selected Best Parameters")
    lines.append("")
    lines.append("| Dataset | Grid Size | EliminationSearchCV | GridSearchCV |")
    lines.append("|---|---|---|---|")

    for r in sorted_recs:
        e = r["elim"]
        g = r["grid"]
        ep = ", ".join(f"{k}={v}" for k, v in sorted(e["params"].items()))
        gp = ", ".join(f"{k}={v}" for k, v in sorted(g["params"].items()))
        lines.append(f"| {r['dataset']} | {r['grid_type']} | `{ep}` | `{gp}` |")

    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def build_speed_summary(all_records: List[Dict]) -> str:
    """Overall speed and score summary grouped by model and grid size."""
    from collections import defaultdict
    summary_data = defaultdict(lambda: {"elim_time": [], "grid_time": [], "acc_diffs": []})

    for r in all_records:
        key = (r["model"], r["grid_type"])
        summary_data[key]["elim_time"].append(r["elim"]["time"])
        summary_data[key]["grid_time"].append(r["grid"]["time"])
        diff = r["elim"]["metrics"]["accuracy"] - r["grid"]["metrics"]["accuracy"]
        summary_data[key]["acc_diffs"].append(diff)

    lines = ["### Speed & Score Summary (Light vs Full Grid Scaling)", ""]
    lines.append("| Model | Grid Size | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |")
    lines.append("|---|---|---|---|---|---|")

    # Order keys by model and then light -> full
    sorted_keys = sorted(summary_data.keys(), key=lambda x: (x[0], x[1] == "Full"))

    for model_name, grid_type in sorted_keys:
        stats = summary_data[(model_name, grid_type)]
        et = stats["elim_time"]
        gt = stats["grid_time"]
        diffs = stats["acc_diffs"]

        grid_config = LIGHT_MODELS if grid_type == "Light" else FULL_MODELS
        combos = grid_total_fits(grid_config[model_name]["param_grid"]) // CV_FOLDS
        avg_speedup = sum(g / e for e, g in zip(et, gt) if e > 0) / len(et)
        avg_diff = sum(diffs) / len(diffs)
        diff_str = f"+{avg_diff:.4f}" if avg_diff >= 0 else f"{avg_diff:.4f}"
        lines.append(
            f"| {model_name} | {grid_type} | "
            f"**{sum(et)/len(et):.2f}s** | "
            f"{sum(gt)/len(gt):.2f}s | "
            f"**{avg_speedup:.1f}x** | "
            f"{diff_str} |"
        )

    lines.append("")
    return "\n".join(lines)

# ---------------------------------------------------------------------------
# Main Runner
# ---------------------------------------------------------------------------

def main():
    print("=" * 80)
    print("EliminationSearchCV vs GridSearchCV -- Light vs Full Grid Scaling")
    print(f"  Version:         {BENCHMARK_VERSION}")
    print(f"  CV Folds:        {CV_FOLDS}")
    print(f"  Elim Rate:       {ELIMINATION_RATE}")
    print(f"  Max Sample Size: {SAMPLE_SIZE}")
    print(f"  Datasets:        {list(DATASETS.keys())}")
    print(f"  Models:          {list(FULL_MODELS.keys())}")
    print("=" * 80)

    all_records = []

    for ds_name, ds_cfg in DATASETS.items():
        print(f"\n[Dataset] {ds_name}")
        X, y, status = load_and_preprocess(ds_cfg)
        if X is None:
            print(f"  Skipped -- {status}")
            continue
        print(f"  Shape: {X.shape}  |  {status}")

        n_orig = len(X)
        if n_orig > SAMPLE_SIZE:
            X, _, y, _ = train_test_split(
                X, y, train_size=SAMPLE_SIZE, random_state=42, stratify=y
            )
            print(f"  Sampled to {SAMPLE_SIZE:,}")

        n_used = len(X)

        for model_name in FULL_MODELS.keys():
            # Run both grid configurations
            for grid_type in ["Light", "Full"]:
                selected_grids = LIGHT_MODELS if grid_type == "Light" else FULL_MODELS
                model_cfg = selected_grids[model_name]
                estimator = model_cfg["estimator"]
                param_grid = model_cfg["param_grid"]
                n_fits = grid_total_fits(param_grid)

                print(f"  [Model] {model_name} ({grid_type} Grid - {n_fits} GridSearchCV fits)")

                try:
                    ep, em, et = run_elimination(estimator, param_grid, X, y)
                    print(f"    Elim  acc={em['accuracy']:.4f}  time={et:.2f}s  params={ep}")
                except Exception as e:
                    print(f"    Elim  FAILED: {e}")
                    ep, em, et = {}, {m: 0.0 for m in EVAL_METRICS}, 0.0

                try:
                    gp, gm, gt = run_grid_search(estimator, param_grid, X, y)
                    print(f"    Grid  acc={gm['accuracy']:.4f}  time={gt:.2f}s  params={gp}")
                except Exception as e:
                    print(f"    Grid  FAILED: {e}")
                    gp, gm, gt = {}, {m: 0.0 for m in EVAL_METRICS}, 0.0

                all_records.append({
                    "dataset": ds_name,
                    "n":       n_used,
                    "model":   model_name,
                    "grid_type": grid_type,
                    "elim":    {"params": ep, "metrics": em, "time": et},
                    "grid":    {"params": gp, "metrics": gm, "time": gt},
                })

    # --- Build Markdown report ---
    print("\n\nBuilding Markdown report...")

    version_str = f"v{BENCHMARK_VERSION}" if BENCHMARK_VERSION and not BENCHMARK_VERSION.startswith("v") else BENCHMARK_VERSION
    version_header = f" ({version_str})" if version_str else ""

    md_lines = [
        f"# Scaling Benchmark Results{version_header} — EliminationSearchCV vs GridSearchCV",
        "",
        "> Compares EliminationSearchCV vs GridSearchCV on both **Light** (small) and **Full** (large) grids.",
    ]
    if BENCHMARK_VERSION:
        md_lines.append(f"> **EliminationSearchCV Package Version:** `{BENCHMARK_VERSION}`  ")
    md_lines += [
        f"> **Settings:** `cv={CV_FOLDS}` · `elimination_rate={ELIMINATION_RATE}` · "
        f"`primary_scoring={PRIMARY_SCORING}` · `sample_size={SAMPLE_SIZE:,}`  ",
        "> **Bold** = winner for that metric. Reproduce with `python benchmarks/benchmark.py`.",
        "",
        "---",
        "",
    ]

    for model_name in FULL_MODELS.keys():
        recs = [r for r in all_records if r["model"] == model_name]
        if not recs:
            continue
        md_lines.append(build_model_table(model_name, recs))
        md_lines.append("---")
        md_lines.append("")

    md_lines.append(build_speed_summary(all_records))
    md_lines.append("---")
    md_lines.append("")

    md_lines += [
        "> Environment: Python 3.12, scikit-learn, Windows 11",
        "> Results may vary slightly on re-runs due to random sampling and CV splits.",
    ]

    md_content = "\n".join(md_lines)

    # Resolution of output filename based on settings parameters
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    rate_str = str(ELIMINATION_RATE).replace(".", "")
    size_str = f"{SAMPLE_SIZE // 1000}k" if (SAMPLE_SIZE >= 1000 and SAMPLE_SIZE % 1000 == 0) else str(SAMPLE_SIZE)
    version_clean = BENCHMARK_VERSION.replace(" ", "_").strip()
    version_suffix = f"v{version_clean}" if not version_clean.startswith("v") else version_clean

    out_filename = f"benchmark_results_scaling_cv{CV_FOLDS}_rate{rate_str}_size{size_str}_{version_suffix}.md"
    history_filename = f"benchmark_results_scaling_cv{CV_FOLDS}_rate{rate_str}_size{size_str}_{version_suffix}_{timestamp}.md"
    
    history_dir = REPO_ROOT / "benchmarks" / "history"
    history_dir.mkdir(parents=True, exist_ok=True)
    history_path = history_dir / history_filename

    # 1. Write report
    out_dir = REPO_ROOT / "benchmarks" / "marks"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / out_filename
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    # 2. Write history copy
    with open(history_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Results written to: {out_path}")
    print(f"History copy saved to: {history_path}")


if __name__ == "__main__":
    main()
