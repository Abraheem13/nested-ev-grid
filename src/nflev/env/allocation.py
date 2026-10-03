"""Feasible charging allocation by construction.

Every dispatch, from any method, ends in the set
    C = { c : 0 <= c_i <= cap_i,  sum_i c_i <= P_cap },
where cap_i = min(charger rating, power that completes vehicle i's remaining
need within the interval).

* `project`      exact Euclidean projection onto C (bisection on the dual of
                 the sum constraint); used for per-vehicle rate actions.
* `llf_allocate` turns an aggregate power set point into per-vehicle rates by
                 least-laxity-first water filling; used for the aggregate
                 actions of the learned agents. With `guard=True` the set point
                 is raised, if necessary, to cover every vehicle whose laxity
                 is below one dispatch interval (deadline guard).
"""
from __future__ import annotations

import numpy as np


def project(c_raw: np.ndarray, cap: np.ndarray, p_cap: float, tol: float = 1e-9) -> np.ndarray:
    c_raw = np.asarray(c_raw, float)
    cap = np.asarray(cap, float)
    c0 = np.clip(c_raw, 0.0, cap)
    if c0.sum() <= p_cap + tol:
        return c0
    lo, hi = 0.0, float(max(c_raw.max(), 0.0))
    for _ in range(200):
        mu = 0.5 * (lo + hi)
        if np.clip(c_raw - mu, 0.0, cap).sum() > p_cap:
            lo = mu
        else:
            hi = mu
        if hi - lo < tol:
            break
    return np.clip(c_raw - hi, 0.0, cap)


def laxity_h(hours_left: np.ndarray, need_kwh: np.ndarray, p_max_kw: float, eff: float) -> np.ndarray:
    """Slack time: hours left minus hours of full-rate charging still needed."""
    return hours_left - need_kwh / (eff * p_max_kw)


def llf_allocate(u: float, cap: np.ndarray, laxity: np.ndarray, p_cap: float,
                 interval_h: float, guard: bool, mode: str = "llf") -> tuple[np.ndarray, float]:
    """Returns (rates, applied_u). u in [0, 1] is the fraction of the available
    power min(P_cap, sum cap) to dispatch. mode="llf" fills vehicles in order of
    increasing laxity; mode="proportional" (ablation) serves guarded vehicles
    first and shares the rest in proportion to cap."""
    cap = np.asarray(cap, float)
    avail = min(p_cap, float(cap.sum()))
    if avail <= 0:
        return np.zeros_like(cap), 0.0
    target = float(np.clip(u, 0.0, 1.0)) * avail
    urgent = laxity <= interval_h
    if guard:
        target = max(target, min(float(cap[urgent].sum()), avail))
    rates = np.zeros_like(cap)
    if mode == "llf":
        rem = target
        for i in np.argsort(laxity, kind="stable"):
            if rem <= 1e-12:
                break
            rates[i] = min(cap[i], rem)
            rem -= rates[i]
    elif mode == "proportional":
        first = urgent if guard else np.zeros_like(urgent)
        rates[first] = cap[first] * min(1.0, target / max(cap[first].sum(), 1e-12))
        rem = target - rates.sum()
        rest = ~first
        if rem > 1e-12 and cap[rest].sum() > 0:
            rates[rest] = cap[rest] * min(1.0, rem / cap[rest].sum())
    else:
        raise ValueError(mode)
    return rates, target / avail
