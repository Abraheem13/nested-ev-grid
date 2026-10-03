"""Hourly Dutch day-ahead prices (EUR/MWh) from the ENTSO-E Transparency Platform,
as bundled in the pinned ev2gym==2.0.0 wheel (see sources.py).

The bundled CSV has two quirks that are handled explicitly and tested:
  * rows after index 76680 repeat 2023-01-01..2023-09-30 in dd/mm/yyyy format
    (identical values) and then continue to 2025-01-07; duplicates are dropped;
  * two hours are missing (2023-12-30 23:00 UTC and 2024-12-30 23:00 UTC) and
    are linearly interpolated.
All times are converted to Central European *standard* time (UTC+1, no DST
shift) so that every day has exactly 24 hourly prices.
"""
from __future__ import annotations

import functools

import numpy as np
import pandas as pd

from .sources import ensure_data

PRICE_FILE = "ev2gym/data/Netherlands_day-ahead-2015-2024.csv"


@functools.lru_cache(maxsize=1)
def load_prices() -> pd.Series:
    root = ensure_data(verbose=False)
    df = pd.read_csv(root / PRICE_FILE)
    raw = df["Datetime (UTC)"]
    t = pd.to_datetime(raw, format="%Y-%m-%d %H:%M:%S", errors="coerce")
    rest = t.isna() & raw.notna()
    t[rest] = pd.to_datetime(raw[rest], format="%d/%m/%Y %H:%M")
    s = pd.Series(df["Price (EUR/MWhe)"].values, index=t).loc[lambda x: x.index.notna()]
    s = s[~s.index.duplicated(keep="first")].sort_index()
    full = pd.date_range(s.index.min(), s.index.max(), freq="h")
    s = s.reindex(full).interpolate(limit=2)
    if s.isna().any():
        raise RuntimeError("unexpected gaps in the price series")
    s.index = s.index + pd.Timedelta(hours=1)     # UTC -> CET standard time
    return s


def price_window(day: pd.Timestamp, start_hour: int, hours: int) -> np.ndarray:
    """Hourly prices for `hours` hours starting at `start_hour` (CET) on `day`."""
    s = load_prices()
    t0 = pd.Timestamp(day).normalize() + pd.Timedelta(hours=start_hour)
    w = s.loc[t0: t0 + pd.Timedelta(hours=hours - 1)]
    if len(w) != hours:
        raise ValueError(f"price window {t0} + {hours} h not available")
    return w.values.astype(float)


def candidate_days(years: list[int], horizon_h: int, start_hour: int) -> list[pd.Timestamp]:
    """Days d in `years` whose window [d+start_hour, d+start_hour+horizon_h) is
    fully covered (the horizon includes a 24 h price look-ahead)."""
    s = load_prices()
    days = []
    for y in years:
        for d in pd.date_range(f"{y}-01-01", f"{y}-12-31", freq="D"):
            t0 = d + pd.Timedelta(hours=start_hour)
            if t0 >= s.index[0] and t0 + pd.Timedelta(hours=horizon_h - 1) <= s.index[-1]:
                days.append(d)
    return days
