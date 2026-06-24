from typing import Dict, List, Tuple
from itertools import combinations, product
import itertools

from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
)
from sklearn.model_selection import KFold, StratifiedKFold, train_test_split


# ---------------------------------------------------------------------------
# Combination generation
# ---------------------------------------------------------------------------

def generate_param_combinations(param_grid: Dict[str, List]) -> List[Dict]:
    """Generate every possible combination of all parameters in the grid.

    Args:
        param_grid: Mapping of parameter names to candidate value lists.
            Example::

                {'C': [0.1, 1], 'penalty': ['l1', 'l2']}

    Returns:
        List of dicts, one per combination. Each dict maps every parameter
        name to a single value.

        Example:

        Input:  {'C': [0.1, 1], 'penalty': ['l1', 'l2']}
        Output: [
                    {'C': 0.1, 'penalty': 'l1'},
                    {'C': 0.1, 'penalty': 'l2'},
                    {'C': 1,   'penalty': 'l1'},
                    {'C': 1,   'penalty': 'l2'},
                ]
    """
    keys = list(param_grid.keys())
    value_lists = list(param_grid.values())
    return [
        dict(zip(keys, combo))
        for combo in itertools.product(*value_lists)
    ]


def generate_param_combinations_with_limit(
    param_grid: Dict[str, List],
    limit: int = 1,
) -> List[Dict]:
    """Generate combinations of exactly ``limit`` parameters at a time.

    Useful for the elimination search, which starts by evaluating each
    parameter in isolation (``limit=1``) before evaluating pairs, triplets, etc.

    Args:
        param_grid: Mapping of parameter names to candidate value lists.
        limit: Number of parameters to include in each combination.
            Must be in ``[1, len(param_grid)]``.

    Returns:
        List of dicts where each dict has exactly ``limit`` keys.

        Example (limit=1):

        Input:  {'C': [0.1, 1], 'penalty': ['l1', 'l2']}
        Output: [
                    {'C': 0.1},
                    {'C': 1},
                    {'penalty': 'l1'},
                    {'penalty': 'l2'},
                ]

        Example (limit=2):

        Input:  {'C': [0.1, 1], 'penalty': ['l1', 'l2']}
        Output: [
                    {'C': 0.1, 'penalty': 'l1'},
                    {'C': 0.1, 'penalty': 'l2'},
                    {'C': 1,   'penalty': 'l1'},
                    {'C': 1,   'penalty': 'l2'},
                ]
    """
    result: List[Dict] = []
    for selected_keys in combinations(param_grid.keys(), limit):
        selected_value_lists = [param_grid[key] for key in selected_keys]
        for value_combo in product(*selected_value_lists):
            result.append(dict(zip(selected_keys, value_combo)))
    return result


# ---------------------------------------------------------------------------
# Scoring
# ---------------------------------------------------------------------------

_SUPPORTED_METRICS = {"accuracy", "precision", "recall", "f1", "roc_auc"}


def get_model_score(
    model,
    X_val,
    y_val,
    scoring: str,
) -> float:
    """Compute a single evaluation metric for a fitted model on validation data.

    Args:
        model: A fitted scikit-learn estimator.
        X_val: Validation features of shape ``(n_samples, n_features)``.
        y_val: True labels of shape ``(n_samples,)``.
        scoring: Metric name. One of ``'accuracy'``, ``'precision'``,
            ``'recall'``, ``'f1'``, ``'roc_auc'``.

    Returns:
        The computed metric score as a float.

    Raises:
        ValueError: If ``scoring`` is not one of the supported metric names.
    """
    y_pred = model.predict(X_val)

    metric_fn = {
        "accuracy":  lambda: accuracy_score(y_val, y_pred),
        "precision": lambda: precision_score(y_val, y_pred),
        "recall":    lambda: recall_score(y_val, y_pred),
        "f1":        lambda: f1_score(y_val, y_pred),
        "roc_auc":   lambda: roc_auc_score(y_val, y_pred),
    }

    if scoring not in metric_fn:
        raise ValueError(
            f"Unsupported scoring '{scoring}'. "
            f"Choose one of: {sorted(_SUPPORTED_METRICS)}"
        )

    return metric_fn[scoring]()


# ---------------------------------------------------------------------------
# Cross-validation fold creation
# ---------------------------------------------------------------------------

def create_cv_data_sets(
    X,
    y,
    cv: int = 5,
    stratified: bool = True,
) -> List[Tuple]:
    """Split data into cross-validation folds and return train/validation pairs.

    Args:
        X: Feature matrix of shape ``(n_samples, n_features)``. Accepts both
            NumPy arrays and pandas DataFrames.
        y: Target vector of shape ``(n_samples,)``. Accepts both NumPy arrays
            and pandas Series.
        cv: Number of folds. Defaults to ``5``.
        stratified: If ``True`` (default), uses ``StratifiedKFold`` to
            preserve the class distribution in each fold. Use ``False`` for
            regression tasks.

    Returns:
        List of ``(X_train, y_train, X_val, y_val)`` tuples, one per fold.

        Example (cv=2):

        Input:  X shape (100, 5), y shape (100,)
        Output: [
                    (X_train_fold1, y_train_fold1, X_val_fold1, y_val_fold1),  # 80 train / 20 val
                    (X_train_fold2, y_train_fold2, X_val_fold2, y_val_fold2),  # 80 train / 20 val
                ]
    """
    splitter = (
        StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
        if stratified
        else KFold(n_splits=cv, shuffle=True, random_state=42)
    )

    # Support both pandas (iloc-based) and NumPy (index-based) slicing.
    is_pandas = hasattr(X, "iloc")

    folds: List[Tuple] = []
    for train_idx, val_idx in splitter.split(X, y):
        if is_pandas:
            folds.append((
                X.iloc[train_idx], y.iloc[train_idx],
                X.iloc[val_idx],   y.iloc[val_idx],
            ))
        else:
            folds.append((
                X[train_idx], y[train_idx],
                X[val_idx],   y[val_idx],
            ))

    return folds
