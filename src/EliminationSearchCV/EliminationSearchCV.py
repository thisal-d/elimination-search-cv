from .Utils import get_model_score, create_cv_data_sets, generate_param_combinations_with_limit
from sklearn.base import clone
from typing import Dict, List


class EliminationSearchCV:
    """A hyperparameter search strategy that progressively eliminates low-scoring
    parameter values across multiple rounds of cross-validated evaluation.

    In each round, parameter combinations of increasing complexity are scored.
    Low-scoring values are pruned from the search grid based on ``reduce_rate``,
    narrowing the search space before the next round begins.

    Args:
        estimator: A scikit-learn compatible estimator (must implement ``fit``
            and ``predict``).
        param_grid: Dictionary mapping parameter names to lists of candidate
            values. Example::

                {
                    'C':        [0.01, 0.1, 1, 10],
                    'penalty':  ['l1', 'l2'],
                    'solver':   ['liblinear', 'saga'],
                    'max_iter': [1000, 2000],
                }

        scoring: Metric name used to evaluate each candidate. Supported values:
            ``'accuracy'``, ``'precision'``, ``'recall'``, ``'f1'``,
            ``'roc_auc'``.
        cv: Number of cross-validation folds. Defaults to ``5``.
        reduce_rate: Fraction of low-scoring values to drop after each round.
            Must be in ``[0.0, 1.0)``. Defaults to ``0.8`` (keep best 20%).

    Attributes:
        best_params_ (Dict[str, Any]): Best parameter combination found after
            fitting, as a flat dict of scalar values ready for
            ``estimator.set_params(**best_params_)``.
    """

    def __init__(
        self,
        estimator,
        param_grid: Dict[str, List],
        scoring: str,
        cv: int = 5,
        reduce_rate: float = 0.8,
    ):
        self.estimator = estimator
        self.param_grid = param_grid
        self.scoring = scoring
        self.cv = cv
        self.reduce_rate = reduce_rate

        # Working copy of the grid — shrinks each round as values are pruned.
        self._active_param_grid: Dict[str, List] = dict(param_grid)

        # Cross-validation fold splits, populated during fit().
        self._folds: List = []

        # Populated after fit() completes.
        self.best_params_: Dict = {}

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def fit(self, X, y) -> "EliminationSearchCV":
        """Run the elimination search over ``X`` and ``y``.

        Iterates from single-parameter combinations up to full combinations
        of all parameters. After each round, low-scoring parameter values are
        pruned from the active grid.

        Args:
            X: Feature matrix of shape ``(n_samples, n_features)``.
            y: Target vector of shape ``(n_samples,)``.

        Returns:
            self: The fitted ``EliminationSearchCV`` instance (sklearn convention).
        """
        self._folds = create_cv_data_sets(X, y, cv=self.cv)
        n_params = len(self.param_grid)
        total_combinations_evaluated = 0

        for round_idx in range(n_params):
            combination_size = round_idx + 1  # 1-param, 2-param, … n-param

            candidates = generate_param_combinations_with_limit(
                self._active_param_grid, limit=combination_size
            )
            scores = self._score_candidates(candidates)
            self._eliminate_low_scoring_values(candidates, scores)

            total_combinations_evaluated += len(candidates)

        # Unwrap each single-element list to a scalar.
        # Before: {'C': [1], 'penalty': ['l1'], 'solver': ['liblinear'], 'max_iter': [1000]}
        # After:  {'C':  1,  'penalty':  'l1',  'solver':  'liblinear',  'max_iter':  1000}
        self.best_params_ = {
            key: values[0] for key, values in self._active_param_grid.items()
        }
        return self

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _eliminate_low_scoring_values(
        self,
        candidates: List[Dict],
        scores: List[float],
    ) -> None:
        """Remove low-scoring parameter values from the active grid.

        **Round 1** (each candidate has exactly 1 key-value pair):
            Each parameter is evaluated independently. The bottom
            ``reduce_rate`` fraction of values are dropped per parameter.

            Example (reduce_rate=0.8, keep best 20%):

            Input  active_grid['C'] = [0.001, 0.01, 0.1, 1, 10, 100]
                   scores            = [0.70,  0.72, 0.75, 0.90, 0.88, 0.85]
            Output active_grid['C'] = [1]   ← top 20% of 6 values = 1 value

        **Later rounds** (each candidate has 2+ key-value pairs):
            All candidates are ranked globally. The top ``(1 - reduce_rate)``
            fraction is kept. Parameters not present in any kept candidate
            retain their current values unchanged.

            Example (reduce_rate=0.8, keep best 20% of 10 combinations):

            Input  candidates = [{C:1, penalty:'l1'}, {C:10, penalty:'l2'}, …]
            Kept   top 2 → unique values extracted per parameter
            Output active_grid updated to only the values seen in top 2 combos

        Args:
            candidates: List of parameter dicts evaluated this round.
            scores: Cross-validated scores parallel to ``candidates``.
        """
        if not candidates:
            return

        if len(candidates[0]) == 1:
            self._eliminate_single_param_values(candidates, scores)
        else:
            self._eliminate_multi_param_values(candidates, scores)

    def _eliminate_single_param_values(
        self,
        candidates: List[Dict],
        scores: List[float],
    ) -> None:
        """Eliminate low-scoring values per-parameter independently (Round 1).

        Builds a score map for each (parameter, value) pair, then keeps only
        the top ``(1 - reduce_rate)`` values for each parameter.

        Args:
            candidates: Single-key dicts, e.g. ``[{'C': 0.1}, {'C': 1}, …]``.
            scores: Score for each candidate, parallel to ``candidates``.
        """
        # Step 1: Collect the best score observed for each (param, value) pair.
        #
        # Before: candidates = [{'C': 0.01}, {'C': 1}, {'penalty': 'l1'}, …]
        #         scores     = [0.72,        0.90,      0.88,             …]
        #
        # After:  param_value_scores = {
        #             'C':       {0.01: 0.72, 1: 0.90},
        #             'penalty': {'l1': 0.88, 'l2': 0.85},
        #             …
        #         }
        param_value_scores: Dict[str, Dict] = {}
        for candidate, score in zip(candidates, scores):
            param_key, param_value = next(iter(candidate.items()))
            param_value_scores.setdefault(param_key, {})
            current_best = param_value_scores[param_key].get(param_value, -1)
            param_value_scores[param_key][param_value] = max(current_best, score)

        # Step 2: For each parameter, sort its values by score and keep the top fraction.
        #
        # Before: param_value_scores['C'] = {0.01: 0.72, 0.1: 0.75, 1: 0.90, 10: 0.88, …}
        #         n_total=6, reduce_rate=0.8 → n_keep = max(1, round(6*0.2)) = 1
        #
        # After:  active_grid['C'] = [1]
        new_grid: Dict[str, List] = {}
        for param_key, value_score_map in param_value_scores.items():
            n_total = len(value_score_map)

            # A parameter with only one value is never pruned.
            if n_total == 1:
                new_grid[param_key] = list(value_score_map.keys())
                continue

            sorted_by_score = sorted(
                value_score_map.items(), key=lambda kv: kv[1], reverse=True
            )
            n_keep = max(1, round(n_total * (1 - self.reduce_rate)))
            new_grid[param_key] = [value for value, _ in sorted_by_score[:n_keep]]

        self._active_param_grid = new_grid

    def _eliminate_multi_param_values(
        self,
        candidates: List[Dict],
        scores: List[float],
    ) -> None:
        """Eliminate low-scoring values by global combination ranking (Rounds 2+).

        Sorts all candidates by score, keeps the top ``(1 - reduce_rate)``
        fraction, and restricts each parameter to the values seen in those
        top candidates. Parameters absent from all top candidates keep their
        current values (they are not zeroed out).

        Args:
            candidates: Multi-key dicts, e.g.
                ``[{'C': 1, 'penalty': 'l1'}, {'C': 10, 'penalty': 'l2'}, …]``.
            scores: Score for each candidate, parallel to ``candidates``.
        """
        n_total = len(candidates)
        n_keep = max(1, round(n_total * (1 - self.reduce_rate)))

        # Step 1: Rank all candidates and keep the top fraction.
        #
        # Before: 10 combinations with scores [0.80, 0.90, 0.75, …]
        # After:  kept_candidates = top 2 combinations (n_keep=2 for reduce_rate=0.8)
        ranked = sorted(zip(candidates, scores), key=lambda cs: cs[1], reverse=True)
        kept_candidates = [candidate for candidate, _ in ranked[:n_keep]]

        # Step 2: Start from the current grid, then restrict params seen in kept combos.
        #
        # Before: active_grid = {'C': [1, 10], 'penalty': ['l1', 'l2'], 'solver': ['liblinear'], …}
        #         kept_candidates = [{'C': 1, 'penalty': 'l1'}, {'C': 10, 'penalty': 'l1'}]
        #
        # After:  active_grid = {'C': [1, 10], 'penalty': ['l1'], 'solver': ['liblinear'], …}
        #         ^^ 'solver' was not in any kept combo → unchanged
        new_grid: Dict[str, List] = {
            key: list(values) for key, values in self._active_param_grid.items()
        }

        # Collect unique values per parameter from the kept candidates only.
        params_in_kept = set(
            param_key
            for candidate in kept_candidates
            for param_key in candidate
        )
        for param_key in params_in_kept:
            seen_values: List = []
            for candidate in kept_candidates:
                value = candidate.get(param_key)
                if value is not None and value not in seen_values:
                    seen_values.append(value)
            if seen_values:
                new_grid[param_key] = seen_values

        self._active_param_grid = new_grid

    def _score_candidates(
        self, candidates: List[Dict]
    ) -> List[float]:
        """Cross-validate each candidate parameter combination and return mean scores.

        Each candidate is evaluated across all CV folds. If a candidate causes
        a fitting error (e.g. incompatible ``solver``/``penalty`` combination),
        it receives a score of ``0.0`` and is naturally eliminated in the next
        pruning step.

        Args:
            candidates: List of parameter dicts to evaluate, e.g.
                ``[{'C': 0.1}, {'C': 1, 'penalty': 'l1'}, …]``.

        Returns:
            List of mean cross-validated scores, one per candidate.
            Invalid combinations receive ``0.0``.
        """
        scores: List[float] = []

        for params in candidates:
            fold_scores: List[float] = []
            combination_is_valid = True

            for X_train, y_train, X_val, y_val in self._folds:
                model = clone(self.estimator)
                model.set_params(**params)

                try:
                    model.fit(X_train, y_train)
                except Exception:
                    # Invalid combination (e.g. penalty='l1' with solver='lbfgs').
                    # Score 0.0 ensures it gets pruned in the next round.
                    combination_is_valid = False
                    break

                fold_scores.append(
                    get_model_score(model, X_val, y_val, self.scoring)
                )

            if combination_is_valid and fold_scores:
                scores.append(sum(fold_scores) / len(fold_scores))
            else:
                scores.append(0.0)

        return scores