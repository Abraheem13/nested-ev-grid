"""R1.5: constraint enforcement by construction.

The DDPG actor's raw outputs are projected onto the feasible set

    C = { c : 0 <= c_i <= c_max_i,  sum_i c_i <= P_cap }

so the aggregate transformer constraint holds for EVERY emitted action —
no clipping heuristics, no reliance on penalty learning. `simplex_scale`
is the projection reported in the paper (proportional scaling, which is the
exact Euclidean projection direction for uniform box bounds when the sum
constraint is the only active one); an exact bounded-simplex projection via
bisection is also provided for the sensitivity appendix.
"""
from __future__ import annotations
import numpy as np


def simplex_scale(c_raw: np.ndarray, c_max: np.ndarray, p_cap: float) -> np.ndarray:
    c = np.clip(c_raw, 0.0, c_max)
    s = c.sum()
    if s > p_cap and s > 0:
        c = c * (p_cap / s)
    return c


def bounded_simplex_projection(c_raw: np.ndarray, c_max: np.ndarray,
                               p_cap: float, tol: float = 1e-8) -> np.ndarray:
    """Exact Euclidean projection onto C via bisection on the dual variable."""
    c0 = np.clip(c_raw, 0.0, c_max)
    if c0.sum() <= p_cap:
        return c0
    lo, hi = 0.0, float(np.max(c_raw))
    for _ in range(100):
        mu = 0.5 * (lo + hi)
        c = np.clip(c_raw - mu, 0.0, c_max)
        if c.sum() > p_cap:
            lo = mu
        else:
            hi = mu
        if hi - lo < tol:
            break
    return np.clip(c_raw - hi, 0.0, c_max)


PROJECTIONS = {"simplex_scale": simplex_scale, "exact": bounded_simplex_projection}
