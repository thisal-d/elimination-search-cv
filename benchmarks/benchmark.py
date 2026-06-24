"""
benchmark.py — EliminationSearchCV vs GridSearchCV
====================================================
Runs a head-to-head comparison across multiple real-world classification
datasets and models. Records accuracy, F1, precision, recall, wall-clock
time, and total model fits. Outputs a GitHub-compatible Markdown report.

Usage:
    python benchmarks/benchmark.py

Reproduce results:
    python benchmarks/benchmark.py > benchmark_output.txt

Requirements:
    pip install scikit-learn pandas numpy
    Datasets in './benchmarks/datasets/'
"""

import time
import os
import sys
import warnings
from pathlib import Path
from typing import Dict, List, Tuple, Any

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import GridSearchCV, cross_val_score, train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------------
# Path setup
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from EliminationSearchCV import EliminationSearchCV


# ---------------------------------------------------------------------------
# Dataset definitions
# ---------------------------------------------------------------------------
DATASET_DIR = REPO_ROOT / "benchmarks" / "datasets"

DATASETS = {
    "Heart Failure": {
        "path": DATASET_DIR / "heart_failure_clinical_records_dataset.csv",
        "target": "DEATH_EVENT",
        "drop": [],
        "sep": ",",
        "categorical": [],
    },
    "Cancer Level": {
        "path": DATASET_DIR / "cancer patient data sets.csv",
        "target": "Level",
        "drop": ["index", "Patient Id"],
        "sep": ",",
        "categorical": [],
    },
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
    "Divorce": {
        "path": DATASET_DIR / "divorce_data.csv",
        "target": "Divorce",
        "drop": [],
        "sep": ";",
        "categorical": [],
    },
}

# ---------------------------------------------------------------------------
# Model + param grid definitions (5 models)
# ---------------------------------------------------------------------------
MODELS = {
    "LogisticRegression": {
        "estimator": LogisticRegression(random_state=42),
        "param_grid": {
            "C":        [0.01, 0.1, 1, 10, 100],
            "penalty":  ["l1", "l2"],
            "solver":   ["liblinear", "saga"],
            "max_iter": [500, 1000],
        },
    },
    "RandomForest": {
        "estimator": RandomForestClassifier(random_state=42, n_jobs=-1),
        "param_grid": {
            "n_estimators":      [50, 100, 200],
            "max_depth":         [None, 5, 10],
            "min_samples_split": [2, 5, 10],
        },
    },
    "DecisionTree": {
        "estimator": DecisionTreeClassifier(random_state=42),
        "param_grid": {
            "max_depth":         [None, 5, 10, 20],
            "min_samples_split": [2, 5, 10],
            "criterion":         ["gini", "entropy"],
        },
    },
    "KNeighbors": {
        "estimator": KNeighborsClassifier(),
        "param_grid": {
            "n_neighbors": [3, 5, 9, 15],
            "weights":     ["uniform", "distance"],
            "metric":      ["euclidean", "manhattan"],
        },
    },
    "GradientBoosting": {
        "estimator": GradientBoostingClassifier(random_state=42),
        "param_grid": {
            "n_estimators":  [50, 100, 200],
            "learning_rate": [0.05, 0.1, 0.2],
            "max_depth":     [3, 5],
        },
    },
}

# ---------------------------------------------------------------------------
# Benchmark settings
# ---------------------------------------------------------------------------
CV_FOLDS     = 1
ELIMINATION_RATE  = 0.8
PRIMARY_SCORING = "accuracy"          # used during search
EVAL_METRICS    = ["accuracy", "f1_macro", "precision_macro", "recall_macro"]
SAMPLE_SIZE  = 20_000


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
        notes.append(f"target label-encoded")
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
    """Evaluate best_params across all EVAL_METRICS using cross_val_score."""
    model = type(estimator)(**{**estimator.get_params(), **best_params})
    results = {}
    for metric in EVAL_METRICS:
        scores = cross_val_score(model, X, y, cv=CV_FOLDS,
                                 scoring=metric, error_score=0.0)
        results[metric] = float(scores.mean())
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
    gs = GridSearchCV(
        estimator=estimator,
        param_grid=param_grid,
        scoring=PRIMARY_SCORING,
        cv=CV_FOLDS,
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

METRIC_LABELS = {
    "accuracy":         "Accuracy",
    "f1_macro":         "F1",
    "precision_macro":  "Precision",
    "recall_macro":     "Recall",
}


def build_model_table(model_name: str, records: List[Dict]) -> str:
    """One table per model. Main columns: Dataset | n | Params Match? | Score Match? | Acc (both) | Time (both) | Speedup.
    Secondary metrics (F1, Precision, Recall) and detailed best parameters are collapsible.
    """
    lines = []
    lines.append(f"### {model_name}")
    lines.append("")

    # Main Table: Comparison Summary
    lines.append(
        "| Dataset | n | Params Match? | Score Match? | Acc (Elim) | Acc (Grid) | Time (Elim) | Time (Grid) | Speedup |"
    )
    lines.append("|---|---|---|---|---|---|---|---|---|")

    for r in records:
        e = r["elim"]
        g = r["grid"]

        def fmt(val):
            return f"{val:.4f}"

        def winner(ev, gv):
            if ev >= gv:
                return f"**{fmt(ev)}**"
            return fmt(ev)

        # Check parameter matches (order-independent key-value check after sorting)
        params_matched = dict(sorted(e["params"].items())) == dict(sorted(g["params"].items()))
        params_match_str = "✅ Match" if params_matched else "❌ Diff"

        # Check score matches
        diff = e['metrics']['accuracy'] - g['metrics']['accuracy']
        if abs(diff) < 1e-9:
            score_match_str = "✅ Equal"
        elif diff < 0:
            score_match_str = f"❌ Lower ({diff:.4f})"
        else:
            score_match_str = f"🔥 Higher (+{diff:.4f})"

        speedup = g["time"] / e["time"] if e["time"] > 0 else 0

        # Bold whichever time is faster
        if e["time"] <= g["time"]:
            time_elim_str = f"**{e['time']:.2f}s**"
            time_grid_str = f"{g['time']:.2f}s"
        else:
            time_elim_str = f"{e['time']:.2f}s"
            time_grid_str = f"**{g['time']:.2f}s**"

        lines.append(
            f"| {r['dataset']} | {r['n']:,} | "
            f"{params_match_str} | "
            f"{score_match_str} | "
            f"{winner(e['metrics']['accuracy'], g['metrics']['accuracy'])} | "
            f"{winner(g['metrics']['accuracy'], e['metrics']['accuracy'])} | "
            f"{time_elim_str} | "
            f"{time_grid_str} | "
            f"**{speedup:.1f}x** |"
        )

    lines.append("")

    # Collapsible Table: Best Params & Secondary Metrics
    lines.append("<details>")
    lines.append("<summary><strong>🔍 View Selected Hyperparameters & Secondary Metrics</strong></summary>")
    lines.append("<br>")
    lines.append("")
    lines.append("#### ⚙️ Selected Best Parameters")
    lines.append("")
    lines.append("| Dataset | EliminationSearchCV | GridSearchCV |")
    lines.append("|---|---|---|")

    for r in records:
        e = r["elim"]
        g = r["grid"]
        # Sort keys alphabetically to guarantee order consistency
        ep = ", ".join(f"{k}={v}" for k, v in sorted(e["params"].items()))
        gp = ", ".join(f"{k}={v}" for k, v in sorted(g["params"].items()))
        lines.append(f"| {r['dataset']} | `{ep}` | `{gp}` |")

    lines.append("")
    lines.append("#### 📊 Secondary Metrics (F1, Precision, Recall)")
    lines.append("")
    lines.append(
        "| Dataset | F1 (Elim) | F1 (Grid) | Prec (Elim) | Prec (Grid) | Rec (Elim) | Rec (Grid) |"
    )
    lines.append("|---|---|---|---|---|---|---|")

    for r in records:
        e = r["elim"]
        g = r["grid"]

        def fmt(val):
            return f"{val:.4f}"

        def winner(ev, gv):
            if ev >= gv:
                return f"**{fmt(ev)}**"
            return fmt(ev)

        lines.append(
            f"| {r['dataset']} | "
            f"{winner(e['metrics']['f1_macro'], g['metrics']['f1_macro'])} | "
            f"{winner(g['metrics']['f1_macro'], e['metrics']['f1_macro'])} | "
            f"{winner(e['metrics']['precision_macro'], g['metrics']['precision_macro'])} | "
            f"{winner(g['metrics']['precision_macro'], e['metrics']['precision_macro'])} | "
            f"{winner(e['metrics']['recall_macro'], g['metrics']['recall_macro'])} | "
            f"{winner(g['metrics']['recall_macro'], e['metrics']['recall_macro'])} |"
        )

    lines.append("")
    lines.append("</details>")
    lines.append("")

    return "\n".join(lines)


def build_speed_summary(all_records: List[Dict]) -> str:
    """Overall speed and score summary across all models."""
    from collections import defaultdict
    model_times = defaultdict(lambda: {"elim": [], "grid": []})
    model_score_diffs = defaultdict(list)

    for r in all_records:
        model_times[r["model"]]["elim"].append(r["elim"]["time"])
        model_times[r["model"]]["grid"].append(r["grid"]["time"])
        diff = r["elim"]["metrics"]["accuracy"] - r["grid"]["metrics"]["accuracy"]
        model_score_diffs[r["model"]].append(diff)

    lines = ["### Speed & Score Summary", ""]
    lines.append("| Model | Grid Combos | Avg Elim Time | Avg Grid Time | Avg Speedup | Avg Acc Diff |")
    lines.append("|---|---|---|---|---|---|")

    for model_name, cfg in MODELS.items():
        if model_name not in model_times:
            continue
        et = model_times[model_name]["elim"]
        gt = model_times[model_name]["grid"]
        diffs = model_score_diffs[model_name]
        combos = grid_total_fits(cfg["param_grid"]) // CV_FOLDS
        avg_speedup = sum(g / e for e, g in zip(et, gt) if e > 0) / len(et)
        avg_diff = sum(diffs) / len(diffs)
        diff_str = f"+{avg_diff:.4f}" if avg_diff >= 0 else f"{avg_diff:.4f}"
        lines.append(
            f"| {model_name} | {combos} | "
            f"**{sum(et)/len(et):.2f}s** | "
            f"{sum(gt)/len(gt):.2f}s | "
            f"**{avg_speedup:.1f}x** | "
            f"{diff_str} |"
        )

    lines.append("")
    return "\n".join(lines)



# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print("=" * 70)
    print("EliminationSearchCV vs GridSearchCV -- Benchmark")
    print(f"cv={CV_FOLDS}, elimination_rate={ELIMINATION_RATE}, ")
    print(f"primary_scoring={PRIMARY_SCORING}, sample_size={SAMPLE_SIZE:,}")
    print("=" * 70)

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

        for model_name, model_cfg in MODELS.items():
            estimator  = model_cfg["estimator"]
            param_grid = model_cfg["param_grid"]
            n_fits     = grid_total_fits(param_grid)

            print(f"  [Model] {model_name}  ({n_fits} GridSearchCV fits)")

            try:
                ep, em, et = run_elimination(estimator, param_grid, X, y)
                print(f"    Elim  acc={em['accuracy']:.4f}  f1={em['f1_macro']:.4f}  "
                      f"time={et:.2f}s  params={ep}")
            except Exception as e:
                print(f"    Elim  FAILED: {e}")
                ep, em, et = {}, {m: 0.0 for m in EVAL_METRICS}, 0.0

            try:
                gp, gm, gt = run_grid_search(estimator, param_grid, X, y)
                print(f"    Grid  acc={gm['accuracy']:.4f}  f1={gm['f1_macro']:.4f}  "
                      f"time={gt:.2f}s  params={gp}")
            except Exception as e:
                print(f"    Grid  FAILED: {e}")
                gp, gm, gt = {}, {m: 0.0 for m in EVAL_METRICS}, 0.0

            all_records.append({
                "dataset": ds_name,
                "n":       n_used,
                "model":   model_name,
                "elim":    {"params": ep, "metrics": em, "time": et},
                "grid":    {"params": gp, "metrics": gm, "time": gt},
            })

    # --- Build Markdown report ---
    print("\n\nBuilding Markdown report...")

    md_lines = [
        "# Benchmark Results — EliminationSearchCV vs GridSearchCV",
        "",
        f"> **Settings:** `cv={CV_FOLDS}` · `elimination_rate={ELIMINATION_RATE}` · "
        f"`primary_scoring={PRIMARY_SCORING}` · `sample_size={SAMPLE_SIZE:,}`  ",
        "> **Bold** = winner for that metric. Reproduce with `python benchmarks/benchmark.py`.",
        "",
        "---",
        "",
    ]

    # One table per model
    for model_name in MODELS:
        recs = [r for r in all_records if r["model"] == model_name]
        if not recs:
            continue
        md_lines.append(build_model_table(model_name, recs))
        md_lines.append("---")
        md_lines.append("")

    # Summary
    md_lines.append(build_speed_summary(all_records))
    md_lines.append("---")
    md_lines.append("")

    md_lines += [
        "> Environment: Python 3.12, scikit-learn, Windows 11",
        "> Results may vary slightly on re-runs due to random sampling and CV splits.",
    ]

    md_content = "\n".join(md_lines)

    out_path = REPO_ROOT / "benchmarks" / "benchmark_results.md"
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(md_content)

    print(f"Results written to: {out_path}")


if __name__ == "__main__":
    main()
