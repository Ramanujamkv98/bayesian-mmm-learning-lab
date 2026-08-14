"""Streamlit caches for deterministic, reusable MMM calculations."""

from __future__ import annotations

import numpy as np
import streamlit as st

from .model_loader import ModelArtifacts
from .prediction import predict_scenario, predict_scenario_row


def _historical_scenario(show: str, season: int, artifacts: ModelArtifacts):
    data = artifacts.data
    return data[(data["Show"] == show) & (data["Season"] == season)].copy().reset_index(drop=True)


@st.cache_data(show_spinner=False, max_entries=64)
def historical_prediction(
    show: str,
    season: int,
    draws: int,
    seed: int,
    artifact_fingerprint: str,
    _artifacts: ModelArtifacts,
):
    del artifact_fingerprint
    scenario = _historical_scenario(show, season, _artifacts)
    return predict_scenario(scenario, _artifacts, draws=draws, seed=seed)


@st.cache_data(show_spinner=False, max_entries=128)
def historical_row_prediction(
    show: str,
    season: int,
    row_index: int,
    draws: int,
    seed: int,
    artifact_fingerprint: str,
    _artifacts: ModelArtifacts,
):
    del artifact_fingerprint
    scenario = _historical_scenario(show, season, _artifacts)
    return predict_scenario_row(scenario, row_index, _artifacts, draws=draws, seed=seed)


@st.cache_data(show_spinner=False, max_entries=16)
def adstock_summary(
    channel_index: int,
    artifact_fingerprint: str,
    _artifacts: ModelArtifacts,
):
    del artifact_fingerprint
    posterior = _artifacts.posterior["alpha"][:, channel_index]
    weeks = np.arange(9)
    curves = posterior[:, None] ** weeks[None, :]
    return {
        "median_alpha": float(np.median(posterior)),
        "alpha_interval": np.quantile(posterior, [.05, .95]),
        "weeks": weeks,
        "curve_interval": np.quantile(curves, [.05, .95], axis=0),
    }


@st.cache_data(show_spinner=False, max_entries=64)
def saturation_curve(
    channel_index: int,
    show_index: int,
    max_spend: float,
    points: int,
    half_override: float | None,
    artifact_fingerprint: str,
    _artifacts: ModelArtifacts,
):
    del artifact_fingerprint
    half_draws = _artifacts.posterior["half_saturation"][:, channel_index]
    beta_draws = _artifacts.posterior["beta_show"][:, show_index, channel_index]
    spend = np.linspace(0, max_spend, points)
    x = spend / _artifacts.spend_scale[channel_index]
    half = half_draws[:, None] if half_override is None else half_override
    curves = _artifacts.y_sd * beta_draws[:, None] * x[None, :] / (x[None, :] + half)
    return spend, np.median(curves, axis=0), np.quantile(curves, [.05, .95], axis=0)
