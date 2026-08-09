from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import streamlit as st

from .model_loader import CHANNELS, MAX_LAG, SPEND_COLUMNS, ModelArtifacts, controls_for
from .transforms import geometric_adstock, hill_saturation, make_lag_tensor
from .performance import timed


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_draw_indices(total: int, draws: int, seed: int) -> np.ndarray:
    return np.random.default_rng(seed).choice(total, size=min(draws, total), replace=False)


def select_draws(total: int, draws: int, seed: int = 42) -> np.ndarray:
    if total <= 0 or draws <= 0:
        raise ValueError("total and draws must be positive")
    return _cached_draw_indices(total, draws, seed)


def _validate(frame: pd.DataFrame, artifacts: ModelArtifacts) -> pd.DataFrame:
    required = {"Show", "Season", "Week_Number", "Episode_Type", "Holiday", "LeadIn_Bonus", *SPEND_COLUMNS}
    missing = required.difference(frame.columns)
    if missing:
        raise ValueError(f"Missing scenario columns: {sorted(missing)}")
    frame = frame.copy().reset_index(drop=True)
    if not np.isfinite(frame[SPEND_COLUMNS].to_numpy(float)).all():
        raise ValueError("Spend must be finite")
    if (frame[SPEND_COLUMNS] < 0).any().any():
        raise ValueError("Spend cannot be negative")
    for show in frame["Show"].astype(str):
        if show not in artifacts.show_lookup:
            raise ValueError(f"Unknown show: {show}")
    labels = frame["Show"].astype(str) + " | Season " + frame["Season"].astype(str)
    unknown = sorted(set(labels).difference(artifacts.show_season_lookup))
    if unknown:
        raise ValueError(f"Unknown show-season: {unknown[0]}")
    return frame


def predict_scenario(
    scenario_data: pd.DataFrame,
    artifacts: ModelArtifacts,
    draws: int = 500,
    interval: float = 0.90,
    include_noise: bool = True,
    seed: int = 42,
    draw_indices: np.ndarray | None = None,
    overrides: dict[str, np.ndarray | float] | None = None,
) -> dict[str, Any]:
    """Vectorized posterior prediction and additive contribution decomposition."""
    if not 0 < interval < 1:
        raise ValueError("interval must be between 0 and 1")
    with timed("scenario prediction"):
        frame = _validate(scenario_data, artifacts)
        ids = select_draws(artifacts.n_samples, draws, seed) if draw_indices is None else np.asarray(draw_indices)
        if np.any((ids < 0) | (ids >= artifacts.n_samples)):
            raise ValueError("draw_indices contains an invalid posterior draw")
        posterior = artifacts.posterior
        overrides = overrides or {}

        alpha, half = _posterior_media_parameters(ids, posterior, overrides)
        spend_scaled = frame[SPEND_COLUMNS].to_numpy(float) / artifacts.spend_scale
        lagged, valid = make_lag_tensor(frame, spend_scaled, max_lag=MAX_LAG)
        adstocked = geometric_adstock(lagged, valid, alpha)
        saturated = hill_saturation(adstocked, half)

        show_idx = frame["Show"].map(artifacts.show_lookup).to_numpy(int)
        labels = frame["Show"].astype(str) + " | Season " + frame["Season"].astype(str)
        season_idx = labels.map(artifacts.show_season_lookup).to_numpy(int)
        controls = controls_for(frame, artifacts)

        baseline_scaled = posterior["intercept_show_season"][ids][:, season_idx]
        media_scaled = saturated * posterior["beta_show"][ids][:, show_idx, :]
        control_scaled = np.einsum("rc,dc->dr", controls, posterior["beta_controls"][ids])

        baseline = artifacts.y_mean + artifacts.y_sd * baseline_scaled
        channel_contributions = artifacts.y_sd * media_scaled
        control_contribution = artifacts.y_sd * control_scaled
        expected = baseline + channel_contributions.sum(axis=2) + control_contribution

        if include_noise:
            rng = np.random.default_rng(seed + 1)
            samples = rng.normal(expected, posterior["sigma"][ids, None] * artifacts.y_sd)
        else:
            samples = expected.copy()
        tail = (1 - interval) / 2
        lower, upper = np.quantile(samples, [tail, 1 - tail], axis=0)
    return {
        "draw_indices": ids,
        "samples": samples,
        "expected_samples": expected,
        "mean": samples.mean(axis=0),
        "median": np.median(samples, axis=0),
        "lower": lower,
        "upper": upper,
        "baseline_samples": baseline,
        "control_contribution_samples": control_contribution,
        "channel_contribution_samples": channel_contributions,
        "channel_contributions": np.median(channel_contributions, axis=0),
        "adstocked": adstocked,
        "saturated": saturated,
        "interval": interval,
    }


def _posterior_media_parameters(
    ids: np.ndarray,
    posterior: dict[str, np.ndarray],
    overrides: dict[str, np.ndarray | float],
) -> tuple[np.ndarray, np.ndarray]:
    alpha = np.asarray(overrides.get("alpha", posterior["alpha"][ids]), float)
    half = np.asarray(overrides.get("half_saturation", posterior["half_saturation"][ids]), float)
    if alpha.ndim == 0:
        alpha = np.full((len(ids), len(CHANNELS)), alpha)
    elif alpha.ndim == 1:
        alpha = np.broadcast_to(alpha, (len(ids), len(CHANNELS)))
    if half.ndim == 0:
        half = np.full((len(ids), len(CHANNELS)), half)
    elif half.ndim == 1:
        half = np.broadcast_to(half, (len(ids), len(CHANNELS)))
    return alpha, half


def predict_scenario_row(
    scenario_data: pd.DataFrame,
    row_index: int,
    artifacts: ModelArtifacts,
    draws: int = 500,
    interval: float = 0.90,
    include_noise: bool = True,
    seed: int = 42,
    draw_indices: np.ndarray | None = None,
    overrides: dict[str, np.ndarray | float] | None = None,
) -> dict[str, Any]:
    """Exact selected-row prediction without evaluating unused scenario rows."""
    if not 0 < interval < 1:
        raise ValueError("interval must be between 0 and 1")
    with timed("selected-row scenario prediction"):
        frame = _validate(scenario_data, artifacts)
        if row_index < 0 or row_index >= len(frame):
            raise ValueError("row_index is outside the scenario")
        ids = select_draws(artifacts.n_samples, draws, seed) if draw_indices is None else np.asarray(draw_indices)
        if np.any((ids < 0) | (ids >= artifacts.n_samples)):
            raise ValueError("draw_indices contains an invalid posterior draw")
        posterior = artifacts.posterior
        alpha, half = _posterior_media_parameters(ids, posterior, overrides or {})

        row = frame.iloc[row_index]
        group_mask = (frame["Show"].to_numpy() == row["Show"]) & (frame["Season"].to_numpy() == row["Season"])
        group_indexes = np.flatnonzero(group_mask)
        positions = np.flatnonzero(group_indexes == row_index)
        if len(positions) != 1:
            raise ValueError("row_index does not map uniquely within its show-season")
        position = int(positions[0])
        lag_indexes = group_indexes[max(0, position - MAX_LAG) : position + 1][::-1]
        lagged_row = (
            frame.loc[lag_indexes, SPEND_COLUMNS].to_numpy(float) / artifacts.spend_scale
        ).T
        lag_numbers = np.arange(len(lag_indexes), dtype=float)
        weights = alpha[:, :, None] ** lag_numbers[None, None, :]
        adstocked = np.einsum("cl,dcl->dc", lagged_row, weights) / weights.sum(axis=2)
        saturated = adstocked / (adstocked + half)

        show_idx = artifacts.show_lookup[str(row["Show"])]
        season_label = f"{row['Show']} | Season {row['Season']}"
        season_idx = artifacts.show_season_lookup[season_label]
        controls = controls_for(frame.iloc[[row_index]], artifacts)[0]
        baseline = artifacts.y_mean + artifacts.y_sd * posterior["intercept_show_season"][ids, season_idx]
        channel_contributions = artifacts.y_sd * saturated * posterior["beta_show"][ids, show_idx, :]
        control_contribution = artifacts.y_sd * np.einsum("c,dc->d", controls, posterior["beta_controls"][ids])
        expected = baseline + channel_contributions.sum(axis=1) + control_contribution

        if include_noise:
            rng = np.random.default_rng(seed + 1)
            standard_noise = rng.normal(size=(len(ids), len(frame)))[:, row_index]
            samples = expected + standard_noise * posterior["sigma"][ids] * artifacts.y_sd
        else:
            samples = expected.copy()
        tail = (1 - interval) / 2
        lower, upper = np.quantile(samples, [tail, 1 - tail])
    return {
        "draw_indices": ids,
        "samples": samples,
        "expected_samples": expected,
        "mean": float(samples.mean()),
        "median": float(np.median(samples)),
        "lower": float(lower),
        "upper": float(upper),
        "baseline_samples": baseline,
        "control_contribution_samples": control_contribution,
        "channel_contribution_samples": channel_contributions,
        "channel_contributions": np.median(channel_contributions, axis=0),
        "adstocked": adstocked,
        "saturated": saturated,
        "interval": interval,
    }


def reproduce_stored_mu(artifacts: ModelArtifacts) -> tuple[np.ndarray, np.ndarray]:
    """Rebuild all 420 expected predictions for validation against stored mu_all."""
    ids = np.arange(artifacts.n_samples)
    result = predict_scenario(
        artifacts.data,
        artifacts,
        draws=artifacts.n_samples,
        include_noise=False,
        draw_indices=ids,
    )
    reconstructed = (result["expected_samples"] - artifacts.y_mean) / artifacts.y_sd
    return reconstructed, artifacts.posterior["mu_all"]
