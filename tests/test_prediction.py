from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mmm.model_loader import load_artifacts
from mmm.prediction import predict_scenario, reproduce_stored_mu, select_draws

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
