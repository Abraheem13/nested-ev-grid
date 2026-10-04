"""Residential base-load shape from Pecan Street (25 homes, 15-min, one year), as
bundled in the pinned ev2gym==2.0.0 wheel (see sources.py).

The 25 household traces are averaged into one aggregate residential profile and
normalised by its 99.5th percentile, giving a dimensionless multiplier that
scales the nominal IEEE feeder loads. Days are matched by day of year.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .sources import ensure_data

LOAD_FILE = "ev2gym/data/residential_loads.csv"
STEPS_PER_DAY = 96


@functools.lru_cache(maxsize=1)
def load_profile() -> np.ndarray:
    """(365, 96) array of normalised aggregate residential load."""
    root = ensure_data(verbose=False)
    raw = pd.read_csv(root / LOAD_FILE, header=None).values
    if raw.shape != (365 * STEPS_PER_DAY, 25):
        raise RuntimeError(f"unexpected load data shape {raw.shape}")
    agg = raw.mean(axis=1)
    agg = agg / np.percentile(agg, 99.5)
    return agg.reshape(365, STEPS_PER_DAY)


def load_window(day_of_year: int, start_hour: int, hours: int) -> np.ndarray:
    """Multiplier at 15-min resolution for `hours` hours from `start_hour` on the
    given day of year (1-based), wrapping around the year end."""
    prof = load_profile().reshape(-1)
    start = ((day_of_year - 1) % 365) * STEPS_PER_DAY + start_hour * 4
    idx = (start + np.arange(hours * 4)) % prof.size
    return prof[idx].astype(float)
