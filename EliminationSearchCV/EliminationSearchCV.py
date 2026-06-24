from .Utils import generate_param_combinations, get_model_score, create_cv_data_sets
from sklearn.model_selection import cross_val_score
from sklearn.base import clone
import numpy as np
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, 
    f1_score, confusion_matrix, roc_auc_score, roc_curve)



class EliminationSearchCV:
    def __init__(
        self,
        estimator ,
        param_grid,
        scoring,
        cv
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
        self.cv = cv
        self.scoring = scoring
        self.folds = []
  
        # self.verbose = verbose
        # self.n_jobs = n_jobs if n_jobs is not None else


    def fit(self, X, y):
        # create data sets with 

        self.folds = create_cv_data_sets(X, y)
        print(self.folds)

        # For each parameter, store the best score
        scores = {}

        # Iterating through the parameters
        for parameter_key in self.param_grid.keys():
            scores[parameter_key] = {} # Initialize here, before iterating values

            for parameter_value in self.param_grid[parameter_key]:
            
                # store each fold score
                fold_scores = []
                
                # get each folded score
                for fold in self.folds:
                    # cloned the default estimator
                    cloned_estimator = clone(self.estimator)
                    # create estimatero with current parameter
                    cloned_estimator.set_params(**{parameter_key:parameter_value})

                    cloned_estimator.fit(fold[0], fold[1])

                    fold_scores.append(
                        get_model_score(
                            cloned_estimator,
                            fold[2],
                            fold[3],
                            self.scoring
                        )
                    )
                scores[parameter_key][parameter_value] = sum(fold_scores)/len(fold_scores)

                
        print(scores)

    
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