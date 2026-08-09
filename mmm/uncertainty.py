from __future__ import annotations

import numpy as np


def interval_summary(samples: np.ndarray, levels=(0.50, 0.80, 0.90, 0.95)) -> dict[float, tuple[np.ndarray, np.ndarray]]:
    samples = np.asarray(samples, float)
    result = {}
    for level in levels:
        if not 0 < level < 1:
            raise ValueError("interval levels must be between 0 and 1")
        tail = (1 - level) / 2
        result[level] = tuple(np.quantile(samples, [tail, 1 - tail], axis=0))
    return result
