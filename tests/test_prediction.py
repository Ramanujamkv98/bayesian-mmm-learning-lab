from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.model_loader import load_artifacts, load_model_artifacts
from mmm.prediction import (
    predict_scenario,
    predict_scenario_row,
    reproduce_stored_mu,
    select_draws,
)

ROOT = Path(__file__).resolve().parents[1]


@pytest.fixture(scope="module")
def artifacts():
    return load_artifacts(ROOT / "model" / "hierarchical_mmm_model_2.nc", ROOT / "data" / "gold_model_dataset.csv")


def test_refactor_reproduces_notebook_predictions(artifacts):
    reconstructed, stored = reproduce_stored_mu(artifacts)
    np.testing.assert_allclose(reconstructed, stored, rtol=1e-10, atol=1e-10)


def test_prediction_is_finite_and_well_shaped(artifacts):
    scenario = artifacts.data.iloc[:15].copy()
    result = predict_scenario(scenario, artifacts, draws=25)
    assert result["samples"].shape == (25, 15)
    assert result["channel_contribution_samples"].shape == (25, 15, 3)
    assert np.isfinite(result["samples"]).all()


def test_zero_media_has_zero_channel_contribution(artifacts):
    scenario = artifacts.data.iloc[:15].copy()
    scenario[["Network_TV_Spend", "Cable_TV_Spend", "Digital_Spend"]] = 0
    result = predict_scenario(scenario, artifacts, draws=20, include_noise=False)
    assert np.allclose(result["channel_contribution_samples"], 0)


def test_invalid_scenario_inputs_are_rejected(artifacts):
    scenario = artifacts.data.iloc[:2].copy()
    scenario.loc[0, "Digital_Spend"] = -1
    with pytest.raises(ValueError, match="negative"):
        predict_scenario(scenario, artifacts, draws=5)
    with pytest.raises(ValueError, match="positive"):
        select_draws(10, 0)


def test_model_loader_is_wrapped_as_cached_resource():
    first = load_model_artifacts()
    second = load_model_artifacts()
    assert callable(load_model_artifacts.clear)
    assert first.fingerprint == second.fingerprint
    assert first.n_samples == second.n_samples == 4_000


def test_posterior_subsampling_is_deterministic(artifacts):
    first = select_draws(artifacts.n_samples, 500, seed=77)
    second = select_draws(artifacts.n_samples, 500, seed=77)
    different = select_draws(artifacts.n_samples, 500, seed=78)
    np.testing.assert_array_equal(first, second)
    assert not np.array_equal(first, different)


@pytest.mark.parametrize("row", [0, 5, 14])
def test_selected_row_engine_matches_original_full_engine(artifacts, row):
    scenario = artifacts.data.iloc[:15].copy().reset_index(drop=True)
    scenario.loc[5, "Digital_Spend"] *= 1.35
    ids = select_draws(artifacts.n_samples, 500, seed=77)
    reference = predict_scenario(scenario, artifacts, draw_indices=ids, seed=77)
    optimized = predict_scenario_row(
        scenario, row, artifacts, draw_indices=ids, seed=77
    )

    np.testing.assert_allclose(
        optimized["expected_samples"],
        reference["expected_samples"][:, row],
        rtol=1e-12,
        atol=1e-8,
    )
    np.testing.assert_allclose(
        optimized["samples"], reference["samples"][:, row], rtol=1e-12, atol=1e-8
    )
    np.testing.assert_allclose(
        optimized["channel_contribution_samples"],
        reference["channel_contribution_samples"][:, row, :],
        rtol=1e-12,
        atol=1e-8,
    )
    assert optimized["lower"] == pytest.approx(reference["lower"][row], abs=1e-8)
    assert optimized["upper"] == pytest.approx(reference["upper"][row], abs=1e-8)


def test_fixed_reference_scenario_has_not_drifted(artifacts):
    scenario = artifacts.data.iloc[:15].copy().reset_index(drop=True)
    ids = select_draws(artifacts.n_samples, 500, seed=77)
    result = predict_scenario_row(
        scenario, 5, artifacts, draw_indices=ids, seed=77
    )
    assert np.median(result["expected_samples"]) == pytest.approx(
        138755.1782637213, abs=1e-6
    )
    assert result["lower"] == pytest.approx(126310.98038310026, abs=1e-6)
    assert result["upper"] == pytest.approx(151166.20937064843, abs=1e-6)
    np.testing.assert_allclose(
        result["channel_contributions"],
        [30206.801451003594, 74247.32036011378, 57219.40144361115],
        rtol=1e-12,
        atol=1e-6,
    )
