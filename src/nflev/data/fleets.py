"""EV fleets.

`residential`  synthetic evening-arrival fleet. Its parameters are modelling
               assumptions (stated in the paper), not fitted to data.
`acn_caltech`, `acn_jpl`
               real charging sessions from ACN-Data (Lee et al., 2019) as
               bundled by SustainGym. Each sampled vehicle keeps a real
               session's arrival time of day, dwell time and delivered energy.
               Sessions from 2019 form the training split; sessions from 2020
               and 2021 form the test split. Weekdays only.

Every fleet returns vehicles in *episode time* (hours since the episode start,
which is `start_hour` local time). Departures are truncated at the episode end
so that every vehicle's service is fully accounted for inside the horizon, and
each energy request is capped at what the charger can physically deliver during
the (truncated) dwell, so a service quality of 1.0 is always attainable.
"""
from __future__ import annotations

import functools
import glob
from dataclasses import dataclass

import numpy as np
import pandas as pd

from .sources import ensure_data

BATTERY_CLASSES_KWH = (40.0, 60.0, 75.0, 100.0)
EPISODE_START_HOUR = {"residential": 12, "acn_caltech": 6, "acn_jpl": 6}


@dataclass
class EVSpec:
    arrival_h: float      # episode time
    departure_h: float    # episode time
    need_kwh: float       # energy to deliver to the battery
    battery_kwh: float
    soc_init: float


def _finalise(arr_wall, dwell, need, rng, start_hour, horizon_h, p_max_kw, eff,
              soc_init=None, battery=None):
    out = []
    for i in range(len(arr_wall)):
        a = (arr_wall[i] - start_hour) % 24.0
        if a >= horizon_h - 0.5:
            continue
        d = min(a + dwell[i], horizon_h - 0.25)
        cap = max(0.0, (d - a) * p_max_kw * eff)
        e = float(min(need[i], cap))
        if e < 0.5:
            continue
        if battery is None:
            ok = [b for b in BATTERY_CLASSES_KWH if e <= 0.85 * b] or [BATTERY_CLASSES_KWH[-1]]
            b = float(rng.choice(ok))
        else:
            b = float(battery[i])
        s0 = float(soc_init[i]) if soc_init is not None else \
            float(np.clip(0.9 - e / b - rng.uniform(0.0, 0.1), 0.05, 0.9))
        s0 = min(s0, 1.0 - e / b)
        out.append(EVSpec(arrival_h=float(a), departure_h=float(d), need_kwh=e,
                          battery_kwh=b, soc_init=s0))
    return out


def sample_residential(n: int, rng: np.random.Generator, horizon_h: int,
                       p_max_kw: float, eff: float, params: dict) -> list[EVSpec]:
    p = params
    evs: list[EVSpec] = []
    while len(evs) < n:
        m = n - len(evs)
        arr = np.clip(rng.normal(p["arrival_mu_h"], p["arrival_sigma_h"], m),
                      p["arrival_min_h"], p["arrival_max_h"])
        dwell = np.clip(rng.normal(p["dwell_mu_h"], p["dwell_sigma_h"], m),
                        p["dwell_min_h"], p["dwell_max_h"])
        s0 = np.clip(rng.normal(p["soc_init_mu"], p["soc_init_sigma"], m), 0.05, 0.9)
        st = np.clip(rng.normal(p["soc_target_mu"], p["soc_target_sigma"], m), s0 + 0.05, 1.0)
        bat = rng.choice(BATTERY_CLASSES_KWH, m)
        need = (st - s0) * bat
        evs += _finalise(arr, dwell, need, rng, EPISODE_START_HOUR["residential"],
                         horizon_h, p_max_kw, eff, soc_init=s0, battery=bat)
    return evs[:n]


@functools.lru_cache(maxsize=4)
def acn_sessions(site: str) -> pd.DataFrame:
    root = ensure_data(verbose=False)
    files = sorted(glob.glob(str(root / f"sustaingym/data/evcharging/acn_data/{site}/*.csv.gz")))
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    a = pd.to_datetime(df["arrival"], utc=True).dt.tz_convert("America/Los_Angeles")
    d = pd.to_datetime(df["departure"], utc=True).dt.tz_convert("America/Los_Angeles")
    out = pd.DataFrame({
        "arrival_wall_h": a.dt.hour + a.dt.minute / 60.0 + a.dt.second / 3600.0,
        "dwell_h": (d - a).dt.total_seconds() / 3600.0,
        "energy_kwh": df["delivered_energy (kWh)"].astype(float),
        "year": a.dt.year, "weekday": a.dt.weekday,
    })
    out = out[(out.weekday < 5) & (out.dwell_h > 0.25) & (out.energy_kwh >= 0.5)]
    return out.reset_index(drop=True)


def sample_acn(site: str, split: str, n: int, rng: np.random.Generator, horizon_h: int,
               p_max_kw: float, eff: float) -> list[EVSpec]:
    df = acn_sessions(site)
    pool = df[df.year == 2019] if split == "train" else df[df.year >= 2020]
    evs: list[EVSpec] = []
    while len(evs) < n:
        rows = pool.iloc[rng.integers(0, len(pool), n - len(evs))]
        evs += _finalise(rows.arrival_wall_h.values, rows.dwell_h.values,
                         rows.energy_kwh.values, rng, EPISODE_START_HOUR[f"acn_{site}"],
                         horizon_h, p_max_kw, eff)
    return evs[:n]
