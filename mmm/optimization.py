from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .model_loader import MAX_LAG, SPEND_COLUMNS, ModelArtifacts, controls_for
from .performance import timed


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

    frame = scenario.copy().reset_index(drop=True)
    if row_index < 0 or row_index >= len(frame):
        raise ValueError("row_index is outside the scenario")
    row = frame.iloc[row_index]
    if (frame[SPEND_COLUMNS] < 0).any().any():
        raise ValueError("Spend cannot be negative")
    group_mask = (
        (frame["Show"].to_numpy() == row["Show"])
        & (frame["Season"].to_numpy() == row["Season"])
    )
    group_indexes = np.flatnonzero(group_mask)
    position = int(np.flatnonzero(group_indexes == row_index)[0])
    lag_indexes = group_indexes[max(0, position - MAX_LAG) : position + 1][::-1]
    lagged_row = (
        frame.loc[lag_indexes, SPEND_COLUMNS].to_numpy(float) / artifacts.spend_scale
    ).T

    posterior = artifacts.posterior
    alpha = posterior["alpha"][ids]
    half = posterior["half_saturation"][ids]
    weights = alpha[:, :, None] ** np.arange(len(lag_indexes))[None, None, :]
    denominator = weights.sum(axis=2)
    if lagged_row.shape[1] > 1:
        historical_numerator = np.einsum(
            "cl,dcl->dc", lagged_row[:, 1:], weights[:, :, 1:]
        )
    else:
        historical_numerator = np.zeros_like(denominator)
    show_idx = artifacts.show_lookup[str(row["Show"])]
    season_idx = artifacts.show_season_lookup[
        f"{row['Show']} | Season {row['Season']}"
    ]
    controls = controls_for(frame.iloc[[row_index]], artifacts)[0]
    fixed_effect = (
        artifacts.y_mean
        + artifacts.y_sd * posterior["intercept_show_season"][ids, season_idx]
        + artifacts.y_sd
        * np.einsum("c,dc->d", controls, posterior["beta_controls"][ids])
    )
    beta = posterior["beta_show"][ids, show_idx, :]

    def objective(allocation: np.ndarray) -> float:
        current_scaled = allocation / artifacts.spend_scale
        adstocked = (historical_numerator + current_scaled[None, :]) / denominator
        saturated = adstocked / (adstocked + half)
        expected = fixed_effect + artifacts.y_sd * np.sum(saturated * beta, axis=1)
        return -float(expected.mean())

    with timed("budget optimization"):
        fitted = minimize(
            objective,
            start,
            method="SLSQP",
            bounds=[(0.0, total_budget)] * len(SPEND_COLUMNS),
            constraints={"type": "eq", "fun": lambda x: float(np.sum(x) - total_budget)},
            options={"maxiter": 80, "ftol": 1e-7},
        )
    return normalize_allocation(fitted.x if fitted.success else start, total_budget)
