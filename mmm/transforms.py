from __future__ import annotations

import numpy as np
import pandas as pd


def make_lag_tensor(
    frame: pd.DataFrame,
    media_values: np.ndarray,
    group_columns: list[str] | tuple[str, ...] = ("Show", "Season"),
    max_lag: int = 8,
) -> tuple[np.ndarray, np.ndarray]:
    """Create the exact row × channel × lag history used by the notebook."""
    if len(frame) != len(media_values):
        raise ValueError("frame and media_values must have the same number of rows")
    lagged = np.zeros((len(frame), media_values.shape[1], max_lag + 1), dtype=float)
    valid = np.zeros((len(frame), max_lag + 1), dtype=float)
    for indexes in frame.groupby(list(group_columns), sort=False).indices.values():
        indexes = np.asarray(indexes, dtype=int)
        for local_week, current_row in enumerate(indexes):
            available = min(local_week, max_lag) + 1
            for lag in range(available):
                lagged[current_row, :, lag] = media_values[indexes[local_week - lag], :]
                valid[current_row, lag] = 1.0
    return lagged, valid


def geometric_adstock(
    lagged: np.ndarray,
    valid_lags: np.ndarray,
    alpha: np.ndarray,
) -> np.ndarray:
    """Normalized geometric adstock; supports draw × channel alphas."""
    alpha = np.asarray(alpha, dtype=float)
    if np.any((alpha < 0) | (alpha > 1)):
        raise ValueError("alpha must be between 0 and 1")
    if alpha.ndim == 1:
        alpha = alpha[None, :]
    lags = np.arange(lagged.shape[2], dtype=float)
    weights = alpha[:, :, None] ** lags[None, None, :]
    numerator = np.einsum("rcl,dcl->drc", lagged, weights)
    denominator = np.einsum("rl,dcl->drc", valid_lags, weights)
    return np.divide(numerator, denominator, out=np.zeros_like(numerator), where=denominator > 0)


def hill_saturation(x: np.ndarray, half_saturation: np.ndarray) -> np.ndarray:
    """The fitted reach curve x/(x+k); the notebook has no free slope parameter."""
    x = np.asarray(x, dtype=float)
    k = np.asarray(half_saturation, dtype=float)
    if np.any(x < 0) or np.any(k <= 0):
        raise ValueError("media must be non-negative and half_saturation must be positive")
    if k.ndim == 1:
        k = k[None, None, :]
    elif k.ndim == 2:
        k = k[:, None, :]
    return np.divide(x, x + k, out=np.zeros_like(x), where=(x + k) > 0)
