from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model_loader import SPEND_COLUMNS, ModelArtifacts
from .prediction import predict_scenario


def normalize_allocation(values: np.ndarray, total_budget: float) -> np.ndarray:
    values = np.clip(np.asarray(values, float), 0, None)
    if total_budget < 0:
        raise ValueError("total_budget cannot be negative")
    if values.sum() == 0:
        return np.full(len(values), total_budget / len(values))
    return values / values.sum() * total_budget


def suggest_allocation(
    scenario: pd.DataFrame,
    row_index: int,
    total_budget: float,
    artifacts: ModelArtifacts,
    draws: int = 250,
    seed: int = 42,
) -> np.ndarray:
    if total_budget <= 0:
        raise ValueError("total_budget must be positive")
    ids = np.random.default_rng(seed).choice(artifacts.n_samples, min(draws, artifacts.n_samples), replace=False)
    start = normalize_allocation(scenario.loc[row_index, SPEND_COLUMNS].to_numpy(float), total_budget)

    def objective(allocation: np.ndarray) -> float:
        trial = scenario.copy()
        trial.loc[row_index, SPEND_COLUMNS] = allocation
        result = predict_scenario(trial, artifacts, draw_indices=ids, include_noise=False)
        return -float(result["expected_samples"][:, row_index].mean())

    fitted = minimize(
        objective,
        start,
        method="SLSQP",
        bounds=[(0.0, total_budget)] * len(SPEND_COLUMNS),
        constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)},
        options={"maxiter": 80, "ftol": 1e-7},
    )
    return normalize_allocation(fitted.x if fitted.success else start, total_budget)
