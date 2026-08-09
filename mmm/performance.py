"""Lightweight development-only timing helpers."""

from __future__ import annotations

import logging
import os
from contextlib import contextmanager
from time import perf_counter

LOGGER = logging.getLogger("mmm.performance")
PROFILE_ENABLED = os.getenv("MMM_PROFILE", "").lower() in {"1", "true", "yes", "on"}


@contextmanager
def timed(label: str):
    """Log elapsed milliseconds only when MMM_PROFILE is enabled."""
    if not PROFILE_ENABLED:
        yield
        return
    start = perf_counter()
    try:
        yield
    finally:
        LOGGER.info("%s: %.2f ms", label, (perf_counter() - start) * 1000)
