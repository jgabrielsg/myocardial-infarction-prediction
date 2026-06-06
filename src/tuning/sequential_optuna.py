"""Sequential (one-at-a-time) hyperparameter optimization with Optuna TPE."""

from __future__ import annotations

from typing import Any, Callable, Dict, List, Optional, Sequence

import optuna

optuna.logging.set_verbosity(optuna.logging.WARNING)


class SequentialOptunaTuner:
    """
    Tune hyperparameters one at a time (coordinate descent style).

    Each parameter gets its own Optuna study with TPE sampler and MedianPruner,
    keeping previously optimized params fixed. This avoids combinatorial grid
    explosion while still exploring the space efficiently.
    """

    def __init__(
        self,
        param_order: Sequence[str],
        suggest_fn: Callable[[optuna.Trial, str], Any],
        objective_fn: Callable[[Dict[str, Any]], float],
        *,
        direction: str = "maximize",
        n_trials_per_param: int = 20,
        seed: int = 42,
    ) -> None:
        self.param_order = list(param_order)
        self.suggest_fn = suggest_fn
        self.objective_fn = objective_fn
        self.direction = direction
        self.n_trials_per_param = n_trials_per_param
        self.seed = seed

    def optimize(self, initial_params: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        best_params: Dict[str, Any] = dict(initial_params or {})
        history: List[Dict[str, Any]] = []

        for param_name in self.param_order:
            print(f"  Tuning param: {param_name} ({self.n_trials_per_param} trials)...")

            def objective(trial: optuna.Trial) -> float:
                params = best_params.copy()
                params[param_name] = self.suggest_fn(trial, param_name)
                return self.objective_fn(params)

            study = optuna.create_study(
                direction=self.direction,
                sampler=optuna.samplers.TPESampler(seed=self.seed),
                pruner=optuna.pruners.MedianPruner(n_startup_trials=5),
            )
            study.optimize(objective, n_trials=self.n_trials_per_param, show_progress_bar=False)

            best_params[param_name] = study.best_params[param_name]
            history.append({
                "param": param_name,
                "best_value": study.best_value,
                "best_param_value": study.best_params[param_name],
            })
            print(f"    -> {param_name}={study.best_params[param_name]} (score={study.best_value:.4f})")

        best_params["_tuning_history"] = history
        best_params["_best_score"] = history[-1]["best_value"] if history else 0.0
        return best_params
