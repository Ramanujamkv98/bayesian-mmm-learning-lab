from pathlib import Path

import numpy as np

from mmm.cache import historical_prediction, saturation_curve
from mmm.model_loader import SPEND_COLUMNS, load_artifacts

ROOT = Path(__file__).resolve().parents[1]


def test_cached_historical_prediction_is_deterministic():
    artifacts = load_artifacts(
        ROOT / "model" / "hierarchical_mmm_model_2.nc",
        ROOT / "data" / "gold_model_dataset.csv",
    )
    first = historical_prediction(
        "Show1", 1, 100, 33, artifacts.fingerprint, artifacts
    )
    second = historical_prediction(
        "Show1", 1, 100, 33, artifacts.fingerprint, artifacts
    )
    np.testing.assert_array_equal(first["samples"], second["samples"])
    np.testing.assert_array_equal(
        first["channel_contribution_samples"],
        second["channel_contribution_samples"],
    )


def test_cached_saturation_curve_matches_full_posterior_formula():
    artifacts = load_artifacts(
        ROOT / "model" / "hierarchical_mmm_model_2.nc",
        ROOT / "data" / "gold_model_dataset.csv",
    )
    channel, show, points = 0, 0, 180
    max_spend = float(artifacts.data[SPEND_COLUMNS[channel]].max() * 1.5)
    spend, median, interval = saturation_curve(
        channel,
        show,
        max_spend,
        points,
        None,
        artifacts.fingerprint,
        artifacts,
    )
    x = spend / artifacts.spend_scale[channel]
    curves = (
        artifacts.y_sd
        * artifacts.posterior["beta_show"][:, show, channel, None]
        * x[None, :]
        / (x[None, :] + artifacts.posterior["half_saturation"][:, channel, None])
    )
    np.testing.assert_allclose(median, np.median(curves, axis=0))
    np.testing.assert_allclose(interval, np.quantile(curves, [.05, .95], axis=0))
