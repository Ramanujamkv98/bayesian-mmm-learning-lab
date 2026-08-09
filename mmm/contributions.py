from __future__ import annotations

import numpy as np
import pandas as pd

from .model_loader import CHANNELS


def contribution_frame(result: dict, scenario: pd.DataFrame, aggregate: bool = False) -> pd.DataFrame:
    channel = np.median(result["channel_contribution_samples"], axis=0)
    values = {
        "Baseline": np.median(result["baseline_samples"], axis=0),
        "Controls": np.median(result["control_contribution_samples"], axis=0),
        **{name: channel[:, i] for i, name in enumerate(CHANNELS)},
    }
    frame = pd.DataFrame(values)
    frame["Week"] = scenario["Week_Number"].to_numpy()
    if aggregate:
        return frame.drop(columns="Week").sum().rename_axis("Component").reset_index(name="Contribution")
    return frame


def safe_percentages(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, float)
    denominator = np.sum(np.abs(values), axis=-1, keepdims=True)
    return np.divide(values, denominator, out=np.zeros_like(values), where=denominator > 0) * 100


def marginal_roas(base_result: dict, new_result: dict, incremental_spend: float) -> np.ndarray:
    if incremental_spend <= 0:
        raise ValueError("incremental_spend must be positive")
    return (new_result["expected_samples"] - base_result["expected_samples"]) / incremental_spend
