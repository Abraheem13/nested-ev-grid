"""Episode specifications: everything exogenous about one simulated day.

An `EpisodeSpec` fixes the price day, the base-load day, the load noise and
forecast-error realisation and the EV fleet. Every method evaluated on the same
(scenario, seed) pair sees an identical spec, which makes the comparisons
paired.

Training episodes draw a random day from the training years; evaluation episode
`j` (seed = seed_base + j) uses the j-th day of a fixed random permutation of the
test-year days, so the 50 evaluation episodes are 50 distinct held-out days.
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from ..data import fleets as F
from ..data.loads import load_window
from ..data.prices import candidate_days, price_window

EVAL_PERMUTATION_SEED = 20240101
VAL_PERMUTATION_SEED = 20230101


@dataclass
class EpisodeSpec:
    day: pd.Timestamp
    fleet: str
    start_hour: int
    horizon_h: int
    prices: np.ndarray            # EUR/MWh, hourly, horizon_h + lookahead
    load_forecast: np.ndarray     # nominal multiplier, 15-min, horizon_h * 4
    load_actual: np.ndarray       # realised multiplier per QSTS step
    evs: list                     # list[F.EVSpec]
    ev_agg: np.ndarray            # aggregator index of every vehicle
    behavior_seed: int
    mods: dict = field(default_factory=dict)


def _days(cfg: dict, split: str, fleet: str, horizon_h: int) -> list[pd.Timestamp]:
    """"train" and "val" partition the training years: `data.val_days` days,
    drawn by a fixed permutation, are held out for model selection and never
    used for training."""
    years = {"train": cfg["data"]["train_years"], "val": cfg["data"]["train_years"],
             "test": cfg["data"]["test_years"], "alt": cfg["data"]["alt_regime_years"]}[split]
    look = cfg["simulation"]["price_lookahead_h"]
    days = candidate_days(years, horizon_h + look, F.EPISODE_START_HOUR[fleet])
    n_val = int(cfg["data"].get("val_days", 0))
    if split in ("train", "val") and n_val > 0:
        held = set(np.random.default_rng(VAL_PERMUTATION_SEED).permutation(len(days))[:n_val].tolist())
        days = [d for i, d in enumerate(days) if (i in held) == (split == "val")]
    return days


def make_episode(cfg: dict, fleet: str, split: str, n_ev: int, seed: int,
                 scenario: dict | None = None, eval_index: int | None = None,
                 horizon_h: int | None = None) -> EpisodeSpec:
    scenario = dict(scenario or {})
    sim = cfg["simulation"]
    horizon_h = horizon_h or sim["episode_hours"]
    look = sim["price_lookahead_h"]
    start = F.EPISODE_START_HOUR[fleet]
    rng = np.random.default_rng(seed)
    days = _days(cfg, split, fleet, horizon_h)
    if eval_index is None:
        day = days[int(rng.integers(len(days)))]
    else:
        perm = np.random.default_rng(EVAL_PERMUTATION_SEED).permutation(len(days))
        day = days[int(perm[eval_index % len(days)])]
    prices = price_window(day, start, horizon_h + look)
    load_fc = load_window(int(day.dayofyear), start, horizon_h)

    steps_per_15 = 900 // sim["resolution_s"]
    n_steps = horizon_h * 3600 // sim["resolution_s"]
    actual = np.repeat(load_fc, steps_per_15)[:n_steps]
    sig_f = float(scenario.get("forecast_error_sigma", 0.0))
    if sig_f > 0:                               # hourly-persistent forecast error
        err_h = np.clip(rng.normal(1.0, sig_f, horizon_h), 0.6, 1.4)
        actual = actual * np.repeat(err_h, 3600 // sim["resolution_s"])[:n_steps]
    actual = actual * rng.normal(1.0, sim["load_noise_sigma"], n_steps)

    rp = cfg["reactive_power"]
    p_max, eff = rp["charger_p_max_kw"], rp["charger_efficiency"]
    if fleet == "residential":
        evs = F.sample_residential(n_ev, rng, horizon_h, p_max, eff, cfg["fleets"]["residential"])
    elif fleet.startswith("acn_"):
        evs = F.sample_acn(fleet[4:], "train" if split in ("train", "val") else "test",
                           n_ev, rng, horizon_h, p_max, eff)
    else:
        raise ValueError(fleet)
    shares = np.asarray(cfg["aggregators"]["shares"], float)
    ev_agg = rng.choice(len(shares), size=len(evs), p=shares / shares.sum())
    return EpisodeSpec(day=day, fleet=fleet, start_hour=start, horizon_h=horizon_h,
                       prices=prices, load_forecast=load_fc, load_actual=actual,
                       evs=evs, ev_agg=ev_agg, behavior_seed=int(rng.integers(2**31)),
                       mods=scenario)
