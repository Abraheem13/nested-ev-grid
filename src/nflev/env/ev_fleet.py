"""EV fleet simulation.

Arrival/departure/energy distributions are calibrated per-dataset via config
(NREL EV Project, ACN-Data/Caltech, ElaadNL). Loader utilities that fit these
parameters from the raw datasets live in nflev.data.loaders; the environment
consumes only the fitted parameters, keeping training decoupled from raw data.
"""
from __future__ import annotations
from dataclasses import dataclass
import numpy as np


@dataclass
class EV:
    idx: int
    aggregator: int
    arrival_h: float
    departure_h: float
    soc: float            # current state of charge [0,1]
    soc_target: float
    battery_kwh: float
    p_max_kw: float
    s_rated_kva: float
    connected: bool = False
    departed: bool = False
    initial_need_kwh: float = 0.0
    # Fishbein heterogeneity (Level 3 slow sub-layer)
    w_cost: float = 5.0
    w_norm: float = 2.0
    bias: float = 0.0
    accepted: bool = True  # current participation decision

    @property
    def energy_needed_kwh(self) -> float:
        return max(0.0, (self.soc_target - self.soc) * self.battery_kwh)


class FleetModel:
    """Samples and steps an EV fleet for one episode."""

    def __init__(self, cfg: dict, dataset_cfg: dict, n_ev: int,
                 n_aggregators: int, rng: np.random.Generator):
        self.cfg, self.ds = cfg, dataset_cfg
        self.n_ev, self.n_agg = n_ev, n_aggregators
        self.rng = rng
        self.evs: list[EV] = []

    def reset(self, agg_shares: np.ndarray | None = None) -> list[EV]:
        d, rng = self.ds, self.rng
        # default: aggregator 5 (weak bus 30) stressed with ~26% of arrivals
        shares = agg_shares if agg_shares is not None else np.array(
            [0.18, 0.18, 0.19, 0.19, 0.26])
        b = self.cfg["level3"]["behavior"]
        self.evs = []
        for i in range(self.n_ev):
            arr = float(np.clip(rng.normal(d["arrival_mu_h"], d["arrival_sigma_h"]),
                                d["arrival_min_h"], d["arrival_max_h"]))
            dur = float(np.clip(rng.normal(d["duration_mu_h"], d["duration_sigma_h"]),
                                d["duration_min_h"], d["duration_max_h"]))
            soc0 = float(np.clip(rng.normal(*d["soc_init"]), 0.05, 0.9))
            soct = float(np.clip(rng.normal(*d["soc_target"]), soc0 + 0.05, 1.0))
            batt = float(rng.choice(d["battery_kwh_classes"]))
            self.evs.append(EV(
                idx=i, aggregator=int(rng.choice(self.n_agg, p=shares)),
                arrival_h=arr, departure_h=min(arr + dur, 47.9),
                soc=soc0, soc_target=soct, battery_kwh=batt,
                p_max_kw=self.cfg["reactive_power"]["charger_p_max_kw"],
                s_rated_kva=self.cfg["reactive_power"]["s_rated_kva"],
                w_cost=float(rng.normal(*b["w_cost"])),
                w_norm=float(rng.normal(*b["w_norm"])),
                bias=float(rng.normal(*b["bias"])),
            ))
            self.evs[-1].initial_need_kwh = self.evs[-1].energy_needed_kwh
        return self.evs

    def step_connections(self, t_h: float) -> None:
        for ev in self.evs:
            if not ev.departed and not ev.connected and t_h >= ev.arrival_h:
                ev.connected = True
            if ev.connected and t_h >= ev.departure_h:
                ev.connected = False
                ev.departed = True

    def apply_charging(self, rates_kw: dict[int, float], dt_h: float,
                       efficiency: float) -> None:
        for idx, p in rates_kw.items():
            ev = self.evs[idx]
            if ev.connected and ev.accepted and p > 0:
                ev.soc = min(1.0, ev.soc + p * efficiency * dt_h / ev.battery_kwh)

    def connected_by_aggregator(self, k: int) -> list[EV]:
        return [e for e in self.evs if e.connected and e.aggregator == k]

    def service_quality(self) -> float:
        """Delivered / requested energy over departed vehicles."""
        req = deliv = 0.0
        for ev in self.evs:
            if ev.departed and ev.initial_need_kwh > 0:
                req += ev.initial_need_kwh
                deliv += ev.initial_need_kwh - ev.energy_needed_kwh
        return deliv / req if req > 0 else 1.0
