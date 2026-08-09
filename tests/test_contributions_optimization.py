from pathlib import Path

import numpy as np
import pytest

from mmm.contributions import safe_percentages
from mmm.model_loader import load_artifacts
from mmm.optimization import normalize_allocation, suggest_allocation

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
