"""Repeatable local benchmark for the interactive MMM paths.

Run from the repository root with: python benchmarks/performance.py
"""

from __future__ import annotations

import sys
from pathlib import Path
from time import perf_counter

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from mmm.cache import saturation_curve  # noqa: E402
from mmm.model_loader import SPEND_COLUMNS, load_artifacts  # noqa: E402
from mmm.optimization import normalize_allocation, suggest_allocation  # noqa: E402
from mmm.prediction import (  # noqa: E402
    predict_scenario,
    predict_scenario_row,
    select_draws,
)


def benchmark(label, function, repeats=10):
    timings = []
    value = None
    for _ in range(repeats):
        start = perf_counter()
        value = function()
        timings.append((perf_counter() - start) * 1000)
    print(
        f"{label:<42} median={np.median(timings):8.2f} ms "
        f"min={min(timings):8.2f} ms"
    )
    return value


def main():
    start = perf_counter()
    artifacts = load_artifacts()
    print(f"{'Uncached model load':<42} {(perf_counter() - start) * 1000:8.2f} ms")
    scenario = artifacts.data.iloc[:15].copy().reset_index(drop=True)
    row = 5
    ids = select_draws(artifacts.n_samples, 500, 77)
    benchmark(
        "Original full 500-draw prediction",
        lambda: predict_scenario(scenario, artifacts, draw_indices=ids, seed=77),
    )
    benchmark(
        "Optimized selected-row prediction",
        lambda: predict_scenario_row(
            scenario, row, artifacts, draw_indices=ids, seed=77
        ),
    )

    trial = scenario.copy()
    trial.loc[row, SPEND_COLUMNS] = normalize_allocation(
        np.array([20, 30, 50]), 250_000
    )
    benchmark(
        "Optimized budget rerun (base + trial)",
        lambda: (
            predict_scenario_row(scenario, row, artifacts, draw_indices=ids, seed=77),
            predict_scenario_row(trial, row, artifacts, draw_indices=ids, seed=77),
        ),
    )
    benchmark(
        "Optimized 250-draw budget optimization",
        lambda: suggest_allocation(scenario, row, 250_000, artifacts),
        repeats=3,
    )

    max_spend = float(artifacts.data[SPEND_COLUMNS[0]].max() * 1.5)
    benchmark(
        "Full-posterior saturation curve (cold)",
        lambda: saturation_curve(
            0, 0, max_spend, 180, None, artifacts.fingerprint, artifacts
        ),
        repeats=1,
    )
    print("Warm saturation and historical results are served by Streamlit's runtime cache.")


if __name__ == "__main__":
    main()
