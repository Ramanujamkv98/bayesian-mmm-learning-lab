from pathlib import Path

import numpy as np
import pytest

from mmm.contributions import marginal_roas, safe_percentages
from mmm.model_loader import load_artifacts
from mmm.optimization import normalize_allocation, suggest_allocation
from mmm.prediction import predict_scenario, predict_scenario_row, select_draws

ROOT = Path(__file__).resolve().parents[1]


def test_percentage_guard_handles_all_zero():
    result = safe_percentages(np.zeros((2, 3)))
    assert np.isfinite(result).all()
    assert (result == 0).all()


def test_budget_normalization_respects_total():
    result = normalize_allocation(np.array([2.0, 3.0, 5.0]), 1_000_000)
    assert result.sum() == pytest.approx(1_000_000)
    assert (result >= 0).all()


def test_budget_optimization_is_feasible():
    artifacts = load_artifacts(ROOT / "model" / "hierarchical_mmm_model_2.nc", ROOT / "data" / "gold_model_dataset.csv")
    scenario = artifacts.data.iloc[:15].copy().reset_index(drop=True)
    result = suggest_allocation(scenario, 5, 250_000, artifacts, draws=20)
    assert result.sum() == pytest.approx(250_000, abs=.01)
    assert (result >= 0).all()


def test_optimized_mroas_matches_full_scenario_reference():
    artifacts = load_artifacts(
        ROOT / "model" / "hierarchical_mmm_model_2.nc",
        ROOT / "data" / "gold_model_dataset.csv",
    )
    scenario = artifacts.data.iloc[:15].copy().reset_index(drop=True)
    row = 5
    ids = select_draws(artifacts.n_samples, 200, seed=77)
    base_full = predict_scenario(
        scenario, artifacts, draw_indices=ids, include_noise=False
    )
    base_row = predict_scenario_row(
        scenario, row, artifacts, draw_indices=ids, include_noise=False
    )
    bump = 10_000.0
    for column in ("Network_TV_Spend", "Cable_TV_Spend", "Digital_Spend"):
        bumped = scenario.copy()
        bumped.loc[row, column] += bump
        bumped_full = predict_scenario(
            bumped, artifacts, draw_indices=ids, include_noise=False
        )
        bumped_row = predict_scenario_row(
            bumped, row, artifacts, draw_indices=ids, include_noise=False
        )
        reference = marginal_roas(base_full, bumped_full, bump)[:, row]
        optimized = marginal_roas(base_row, bumped_row, bump)
        np.testing.assert_allclose(optimized, reference, rtol=1e-12, atol=1e-10)
