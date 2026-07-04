"""Non-learning baselines: uncoordinated charging and static TOU pricing.

Both run through the SAME environment (full AC power flow, Fishbein behavior).
Q-control can be enabled or disabled via env scenario_mods["disable_q_control"]
so each baseline is evaluated in its native form (no V2G support), matching v1.
"""
from __future__ import annotations
import numpy as np


class UncoordinatedPolicy:
    """Every connected EV charges at full rate immediately; flat retail price."""
    name = "uncoordinated"

    def __init__(self, retail_price: float = 0.12):
        self.retail = retail_price

    def act(self, env):
        env.set_corridor(self.retail, self.retail + 0.02)
        for k in range(env.n_agg):
            n = len(env.fleet.connected_by_aggregator(k))
            env.set_dispatch(k, np.full(max(n, 1), 999.0), price_frac=0.0)
            # 999 -> clipped to c_max then projected onto transformer cap


class TOUPolicy:
    """Static three-tier time-of-use tariff (0.08/0.12/0.20 $/kWh).
    EVs charge at full rate whenever the Fishbein layer accepts the price."""
    name = "tou"

    def __init__(self, off=0.08, mid=0.12, peak=0.20,
                 peak_hours=(17, 21), mid_hours=(7, 17)):
        self.off, self.mid, self.peak = off, mid, peak
        self.ph, self.mh = peak_hours, mid_hours

    def price(self, t_h: float) -> float:
        h = t_h % 24
        if self.ph[0] <= h < self.ph[1]:
            return self.peak
        if self.mh[0] <= h < self.mh[1]:
            return self.mid
        return self.off

    def act(self, env):
        p = self.price(env.t_s / 3600.0)
        env.set_corridor(p, p + 0.02)
        for k in range(env.n_agg):
            n = len(env.fleet.connected_by_aggregator(k))
            env.set_dispatch(k, np.full(max(n, 1), 999.0), price_frac=0.0)


def run_baseline_episode(env, policy, price_profile, load_profile) -> dict:
    env.reset(price_profile, load_profile)
    n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
    for _ in range(n_intervals):
        policy.act(env)
        env.run_dispatch_interval()
    return env.episode_metrics()