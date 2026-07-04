"""Level 3 slow sub-layer: Fishbein user acceptance model.

R1.4 fix: v1 conflated behavioral response with second-scale reactive control
inside one "continuous" level. v2 splits Level 3 into two non-parametric
sub-layers at distinct characteristic frequencies:

  L3a (this module): behavioral intention, evaluated only at dispatch
      boundaries (15 min) when the effective price can change.
  L3b (qcontrol.py): electromagnetic-timescale reactive correction, realized
      within each QSTS step.

Human price response therefore never co-evolves with voltage transients.
"""
from __future__ import annotations
import numpy as np


class FishbeinBehavior:
    def __init__(self, lambda_ref: float, rng: np.random.Generator):
        self.lambda_ref = lambda_ref
        self.rng = rng
        self.prev_accept_rate = 0.7  # social-norm bootstrap

    def evaluate(self, evs: list, price: float, t_h: float | None = None) -> float:
        """Update per-EV `accepted` flags at a dispatch boundary.
        Returns realized acceptance rate (feeds A_bar for the next interval)."""
        connected = [e for e in evs if e.connected]
        if not connected:
            return self.prev_accept_rate
        # social-norm floor: prevents a self-reinforcing acceptance collapse
        # (A_bar -> 0) that has no behavioral basis for captive home charging
        a_bar = max(0.3, self.prev_accept_rate)
        n_acc = 0
        for ev in connected:
            z = (ev.w_cost * (self.lambda_ref - price) / self.lambda_ref
                 + ev.w_norm * a_bar + ev.bias)
            p_acc = 1.0 / (1.0 + np.exp(-z))
            ev.accepted = bool(self.rng.random() < p_acc)
            # deadline dominance: a driver whose remaining dwell barely covers
            # the remaining energy need accepts regardless of price (attitude
            # toward departure readiness outweighs cost in the Fishbein sum)
            hrs_left = ev.departure_h - t_h if t_h is not None else 99.0
            if ev.energy_needed_kwh > 0.7 * ev.p_max_kw * max(0.0, hrs_left):
                ev.accepted = True
            n_acc += ev.accepted
        self.prev_accept_rate = n_acc / len(connected)
        return self.prev_accept_rate