from typing import Dict, List
import itertools
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve)
from sklearn.model_selection import KFold, StratifiedKFold
from sklearn.model_selection import train_test_split
from itertools import combinations, product
from typing import Dict, List


def generate_param_combinations(param_grid: Dict) -> List[Dict]:
    keys = param_grid.keys()
    values = param_grid.values()
    combinations = list(itertools.product(*values))
    return [dict(zip(keys, combination)) for combination in combinations]



def generate_param_combinations_with_limit(
    param_grid: Dict,
    limit: int = 1
) -> List[Dict]:

    result = []

    for selected_keys in combinations(param_grid.keys(), limit):
        selected_values = [param_grid[key] for key in selected_keys]

        for values in product(*selected_values):
            result.append(dict(zip(selected_keys, values)))

    return result


def train_model(model, train_X, train_y):
    model.fit(train_X, train_y)


def get_model_score(model, test_X, test_y, scoring):
    if scoring == "accuracy":
        return accuracy_score(model.predict(test_X), test_y)

    if scoring == "precision":
        return precision_score(model.predict(test_X), test_y)

    if scoring == "recall":
        return recall_score(model.predict(test_X), test_y)

    if scoring == "f1":
        return f1_score(model.predict(test_X), test_y)

    if scoring == "roc_auc":
        return roc_auc_score(model.predict(test_X), test_y)


def split_data(X, y, test_size=0.2, random_state=42):
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def create_cv_data_sets(train_X, train_y, cv=5, stratified=True):
    # Use StratifiedKFold for classification tasks (like Hypertension prediction)
    if stratified:
        kf = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    else:
        kf = KFold(n_splits=cv, shuffle=True, random_state=42)
        
    folds = []
    
    # Check if inputs are pandas DataFrame/Series
    is_pandas = hasattr(train_X, "iloc")
    
    # kf.split returns the index positions for training and validation splits
    for train_idx, val_idx in kf.split(train_X, train_y):
        if is_pandas:
            X_train_fold = train_X.iloc[train_idx]
            y_train_fold = train_y.iloc[train_idx]
            X_val_fold = train_X.iloc[val_idx]
            y_val_fold = train_y.iloc[val_idx]
        else:
            X_train_fold = train_X[train_idx]
            y_train_fold = train_y[train_idx]
            X_val_fold = train_X[val_idx]
            y_val_fold = train_y[val_idx]
            
        folds.append((X_train_fold, y_train_fold, X_val_fold, y_val_fold))
        
    return folds
