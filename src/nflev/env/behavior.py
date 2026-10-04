"""Level 3a: price acceptance (Fishbein-Ajzen attitude model), evaluated once per
dispatch interval, i.e. at the timescale at which the retail price can change.

    P_accept,i = sigmoid( w_cost,i (lambda_ref - p_i) / lambda_ref + w_norm,i A_bar + b_i )

p_i is the execution price of the vehicle's own aggregator and A_bar the
fleet acceptance rate of the previous interval (floored at `social_floor`).
A vehicle whose remaining need exceeds `deadline_override` x (charger rating x
hours left) accepts regardless of price.
"""
from __future__ import annotations

import numpy as np


class FishbeinBehavior:
    def __init__(self, cfg: dict, n_ev: int, seed: int):
        b = cfg["behavior"]
        self.lambda_ref = b["lambda_ref"]
        self.floor = b["social_floor"]
        self.override = b["deadline_override"]
        rng = np.random.default_rng(seed)
        self.w_cost = rng.normal(*b["w_cost"], n_ev)
        self.w_norm = rng.normal(*b["w_norm"], n_ev)
        self.bias = rng.normal(*b["bias"], n_ev)
        self.rng = np.random.default_rng(seed + 1)
        self.prev_rate = 0.7

    def evaluate(self, idx: np.ndarray, price: np.ndarray, need_kwh: np.ndarray,
                 hours_left: np.ndarray, p_max_kw: float) -> np.ndarray:
        """idx: connected vehicle indices; price: their execution prices.
        Returns a boolean acceptance array aligned with idx."""
        if len(idx) == 0:
            return np.zeros(0, bool)
        a_bar = max(self.floor, self.prev_rate)
        z = (self.w_cost[idx] * (self.lambda_ref - price) / self.lambda_ref
             + self.w_norm[idx] * a_bar + self.bias[idx])
        acc = self.rng.random(len(idx)) < 1.0 / (1.0 + np.exp(-z))
        acc |= need_kwh > self.override * p_max_kw * np.maximum(hours_left, 0.0)
        self.prev_rate = float(acc.mean())
        return acc
