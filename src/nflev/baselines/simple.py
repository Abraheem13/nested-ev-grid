"""Rule-based baselines. All of them dispatch per-vehicle rates through the same
feasible projection as every other method, and can be run with or without the
Level-3 reactive controller (environment flag).

Uncoordinated  every vehicle charges at full rate from arrival until done.
TOU timer      three-tier time-of-use tariff; vehicles charge at full rate in
               the off-peak window (23:00-07:00) and earlier only when their
               laxity is exhausted (the usual behaviour of timer charging).
Price-aware    each vehicle charges in the cheapest dispatch intervals that
               still cover its need before departure. Prices are known only
               `price_lookahead_h` ahead (the window the learned agents observe);
               later intervals are valued at the mean known price (naive
               forecast). Re-planned every interval, so declined or curtailed
               energy is recovered later.
"""
from __future__ import annotations

import numpy as np

from ..env.allocation import laxity_h


class Policy:
    name = "policy"
    sets_corridor = False

    def reset(self, env) -> None:
        pass

    def act(self, env) -> None:
        raise NotImplementedError


def _flat_corridor(env):
    env.set_corridor(env.flat_price, env.flat_price + env.min_width)


class Uncoordinated(Policy):
    name = "uncoordinated"

    def act(self, env):
        _flat_corridor(env)
        for k in range(env.n_agg):
            mem = np.flatnonzero(env.connected & (env.agg == k))
            env.set_rates(k, {int(i): env.p_max for i in mem}, 0.0)


class TOUTimer(Policy):
    name = "tou"
    PEAK, MID, OFF = 0.30, 0.20, 0.12
    PEAK_H, OFF_START, OFF_END = (17, 21), 23, 7

    def price(self, h: float) -> float:
        if self.PEAK_H[0] <= h < self.PEAK_H[1]:
            return self.PEAK
        if h >= self.OFF_START or h < self.OFF_END:
            return self.OFF
        return self.MID

    def act(self, env):
        h = env.wall_hour()
        p = self.price(h)
        env.set_corridor(p, p + env.min_width)
        off = h >= self.OFF_START or h < self.OFF_END
        need = env.need()
        lax = laxity_h(env.hours_left(), need, env.p_max, env.eff)
        for k in range(env.n_agg):
            mem = np.flatnonzero(env.connected & (env.agg == k))
            env.set_rates(k, {int(i): env.p_max for i in mem
                              if off or lax[i] <= env.interval_h}, 0.0)


class PriceAware(Policy):
    name = "price_aware"

    def act(self, env):
        _flat_corridor(env)
        for k in range(env.n_agg):
            env.set_rates(k, {int(i): env.p_max for i in env.price_plan(k)}, 0.0)
