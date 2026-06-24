from .Utils import generate_param_combinations, get_model_score, create_cv_data_sets, generate_param_combinations_with_limit
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
from typing import Dict, List, Tuple
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve)



class EliminationSearchCV:
    def __init__(
        self,
        estimator ,
        param_grid,
        scoring,
        cv,
        reduce_rate: float = 0.8,
        # n_jobs,
        # refit,`1`
        # cv,
        # verbose,
        # pre_dispatch,
        # error_score,
        # return_train_score,
    ):

        self.estimator = estimator
        self.param_grid = param_grid
        self._param_grid_copy = self.param_grid
        self._param_grid_count = {key: len(self.param_grid[key]) for key in self.param_grid.keys()}
        self.cv = cv
        self.scoring = scoring
        self.reduce_rate = reduce_rate
        self.folds = []
        self.best_params_ = {}
  
        # self.verbose = verbose
        # self.n_jobs = n_jobs if n_jobs is not None else


    def fit(self, X, y):

        # create data sets with 
        self.folds = create_cv_data_sets(X, y)
        total_test = 0
        for i in range(len(self.param_grid.keys())):
            print("updated :",self._param_grid_copy)
            all_posible_combination = generate_param_combinations_with_limit(self._param_grid_copy, limit=i+1)
            print("all posibble combinations:", len(all_posible_combination))
            all_scores = self._get_parameter_scores(all_posible_combination)
            self.remove_params_scored_low(all_posible_combination, all_scores)
            total_test += len(all_posible_combination)
            print("----------------------------------------------------\n")
        
        print("total_test", total_test)
        print("best params", self._param_grid_copy)
        # Unwrap each param's list to a scalar (the single best value)
        # e.g. {'C': [1], 'penalty': ['l1']} → {'C': 1, 'penalty': 'l1'}
        self.best_params_ = {key: vals[0] for key, vals in self._param_grid_copy.items()}
        return self.best_params_
        # get all parameters scores As one
            

    def remove_params_scored_low(self, combinations, combination_score):
        """
        Prunes low-scoring parameter values from _param_grid_copy.

        - Round 1 (single-param combos, len == 1):
            For each parameter independently, collect all its values and their
            associated scores, then keep only the top (1 - reduce_rate) fraction.
            Parameters with only 1 value are never pruned.
            E.g. reduce_rate=0.8 → keep best 20% of values per parameter.

        - Later rounds (multi-param combos, len > 1):
            Score each full combination globally and keep the top
            (1 - reduce_rate) fraction, rebuilding _param_grid_copy from those.
            Parameters not covered by any kept combo retain their current values.
        """

        # Guard: nothing to prune if combinations is empty
        if not combinations:
            return

        if len(combinations[0]) == 1:
            # --- Round 1: per-parameter pruning ---

            # Build: { param_key: { param_value: best_score } }
            param_value_scores: Dict[str, Dict] = {}

            for combo, score in zip(combinations, combination_score):
                # Each combo has exactly one key-value pair, e.g. {'C': 0.1}
                param_key, param_value = next(iter(combo.items()))

                if param_key not in param_value_scores:
                    param_value_scores[param_key] = {}

                # Keep the best score seen for this (param, value) pair
                if param_value not in param_value_scores[param_key]:
                    param_value_scores[param_key][param_value] = score
                else:
                    param_value_scores[param_key][param_value] = max(
                        param_value_scores[param_key][param_value], score
                    )

            # For each parameter, keep only the top (1 - reduce_rate) values
            new_param_grid = {}
            for param_key, value_score_map in param_value_scores.items():
                n_total = len(value_score_map)

                # If only 1 value exists, never prune it
                if n_total == 1:
                    new_param_grid[param_key] = list(value_score_map.keys())
                    continue

                # Sort values by score descending
                sorted_values = sorted(
                    value_score_map.items(),
                    key=lambda x: x[1],
                    reverse=True
                )

                # Keep at least 1 value even if reduce_rate is very high
                n_keep = max(1, round(n_total * (1 - self.reduce_rate)))
                new_param_grid[param_key] = [v for v, _ in sorted_values[:n_keep]]

            self._param_grid_copy = new_param_grid

        else:
            # --- Later rounds: global combination pruning ---

            n_total = len(combinations)
            n_keep = max(1, round(n_total * (1 - self.reduce_rate)))

            # Sort combinations by score descending, keep top n_keep
            scored_combos = sorted(
                zip(combinations, combination_score),
                key=lambda x: x[1],
                reverse=True
            )
            kept_combos = [combo for combo, _ in scored_combos[:n_keep]]

            # Start from a copy of the current grid as the baseline.
            # For params that appear in kept combos → restrict to those values only.
            # For params NOT covered by any kept combo → leave their values unchanged.
            new_param_grid: Dict[str, List] = {
                key: list(vals) for key, vals in self._param_grid_copy.items()
            }

            # Find which params actually appear in kept combos
            params_in_kept_combos = set()
            for combo in kept_combos:
                params_in_kept_combos.update(combo.keys())

            # Restrict only the params that appear in kept combos
            for param_key in params_in_kept_combos:
                values_in_kept = []
                for combo in kept_combos:
                    if param_key in combo:
                        val = combo[param_key]
                        if val not in values_in_kept:
                            values_in_kept.append(val)

                # Only update if we end up with at least 1 value
                if values_in_kept:
                    new_param_grid[param_key] = values_in_kept

            self._param_grid_copy = new_param_grid


    
    def _get_parameter_scores(self, param_combination_list: List[Dict]) -> List[float | int]:
        
        # For each parameter, store the best score
        scores = []

        # Iterating through the parameters
        for param_combination in param_combination_list:
            # store each fold score
            fold_scores = []
            invalid = False
            
            # get each folded score
            for fold in self.folds:
                # clone the default estimator
                cloned_estimator = clone(self.estimator)
                # set current parameter combination
                cloned_estimator.set_params(**param_combination)

                try:
                    cloned_estimator.fit(fold[0], fold[1])
                except (ValueError, Exception):
                    # Invalid combination (e.g. penalty='l1' with solver='lbfgs')
                    # Assign score 0 so it gets eliminated naturally
                    invalid = True
                    break

                fold_scores.append(
                    get_model_score(
                        cloned_estimator,
                        fold[2],
                        fold[3],
                        self.scoring
                    )
                )

            if invalid or not fold_scores:
                scores.append(0.0)
            else:
                scores.append(sum(fold_scores) / len(fold_scores))

        return scores


    def _conver_all_parameter_to_one_list_parameter_combination(self, param_grid: Dict) -> List:
        parameter_list = []
        for key, values in param_grid.items():
            for value in values:
                parameter_list.append({key: value})
        return parameter_list
    
    def first_step_get_all_parameters_scores(self):
        params_dict = self._conver_all_parameter_to_one_list_parameter_combination(self.param_grid)
        all_scores = self._get_parameter_scores(params_dict)
        return all_scores
        
        


    
    # def log(self):
    #     # Used for debugging
    #     if self.verbose > 0:
    #         print("self.param_combinations: ", self.param_combinations_)

    # def fit(self, X, y):
    #     n_combinations = len(self.param_combinations_)

       
    #     if self.verbose > 0:
    #         print(f"Fitting {self.cv if self.cv is not None else 5} folds for each of {n_combinations} candidates, totalling {n_combinations * (self.cv if self.cv is not None else 5)} fits")

    #     for idx, params in enumerate(self.param_combinations_):
    #         if self.verbose > 0:
    #             print(f"[{idx+1}/{n_combinations}] Evaluating: {params}")
            
    #         cloned_estimator = clone(self.estimator)
    #         cloned_estimator.set_params(**params)
            
    #         scores = cross_val_score(
    #             cloned_estimator, 
    #             X, 
    #             y, 
    #             cv=self.cv, 
    #             scoring=self.scoring, 
    #             n_jobs=self.n_jobs
    #         )
    #         mean_score = np.mean(scores)
            
    #         if self.verbose > 0:
    #             print(f"Mean score: {mean_score:.4f}")
                
    #         if mean_score > self.best_score_:
    #             self.best_score_ = mean_score
    #             self.best_params_ = params
    #             self.best_estimator_ = cloned_estimator

    #     # Refit best estimator on full data
    #     if self.best_estimator_ is not None:
    #         self.best_estimator_.fit(X, y)

    #     return self