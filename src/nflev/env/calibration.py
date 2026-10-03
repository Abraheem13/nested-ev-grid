"""Base-load calibration (explicit nominal operating point).

The nominal IEEE loads are multiplied by `scale x m(t)`, where m(t) is the
normalised Pecan Street residential profile. `scale` is chosen as the largest
value (on a 0.01 grid) for which the feeder WITHOUT any EVs keeps every bus at
or above V_min + 0.005 p.u. at every 15-min point of the year, with the
substation at `substation_vm_pu`. Any violation in the experiments is therefore
caused by EV charging, not by the base case.
"""
from __future__ import annotations

import functools
import json
import pathlib

import numpy as np

from ..data.loads import load_profile
from ..grid.network import load_network
from ..grid.powerflow import RadialPowerFlow

ROOT = pathlib.Path(__file__).resolve().parents[3]
MARGIN = 0.005


def _vmin_year(pf: RadialPowerFlow, p: np.ndarray, q: np.ndarray, prof: np.ndarray) -> float:
    vmin = np.inf
    for m in np.sort(prof)[::-1][:200]:          # the 200 highest-load points bound the minimum
        vmin = min(vmin, pf.solve(p * m, q * m).vmin)
    return vmin


@functools.lru_cache(maxsize=4)
def _calibrate(network: str, v0: float, v_floor: float) -> float:
    net = load_network(network)
    pf = RadialPowerFlow(net, v0=v0)
    prof = load_profile().reshape(-1)
    lo, hi = 0.1, 2.0
    for _ in range(40):
        mid = 0.5 * (lo + hi)
        if _vmin_year(pf, net.p_load_mw * mid, net.q_load_mvar * mid, prof) >= v_floor:
            lo = mid
        else:
            hi = mid
    return float(np.floor(lo * 100) / 100)


def base_load_scale(cfg: dict, network: str) -> float:
    return _calibrate(network, cfg["network"]["substation_vm_pu"],
                      cfg["voltage"]["v_min"] + MARGIN)


def write_report(cfg: dict, path: pathlib.Path) -> dict:
    out = {}
    for name in ("ieee33", "ieee69"):
        s = base_load_scale(cfg, name)
        net = load_network(name)
        pf = RadialPowerFlow(net, v0=cfg["network"]["substation_vm_pu"])
        prof = load_profile().reshape(-1)
        out[name] = {"base_load_scale": s,
                     "no_ev_min_voltage_pu": _vmin_year(pf, net.p_load_mw * s, net.q_load_mvar * s, prof),
                     "peak_load_mw": float(net.p_load_mw.sum() * s * prof.max())}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(out, indent=2))
    return out
