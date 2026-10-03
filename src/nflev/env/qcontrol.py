"""Level 3b: within-step reactive-power correction with curtailment fallback.

Runs inside one QSTS step, after the power flow:

1. Activation: if min_n |V_n| < V_target = V_min + delta.
2. Capacity: every connected charger can supply Q_i^max = sqrt(S_i^2 - P_i^2)
   (inverter apparent-power rating), so a charger at P_i = 0 offers its full
   rating; aggregate capacity per aggregator bus is the sum over its vehicles.
3. First iteration (proportional): each aggregator bus k injects
   Q_k = Qavail_k * clip((V_target - V_min,sys) / (V_target - V_crit), 0, 1)
         * n_active * w_k / sum(w),     w_k = max(1e-3, V_target - |V_bk|).
4. Later iterations (measured-sensitivity Newton): sigma = dVmin / dQ_total
   measured between the last two power flows; total increment
   1.2 (V_target - V_min,sys) / sigma, split by w_k, capped by Qavail_k.
5. Fallback: if no Q is left, shed `curtailment_step * w_k / max(w)` of the
   EV power at every aggregator bus (most depressed bus sheds the most),
   which also frees inverter capacity for reactive power.
6. Re-solve the power flow; repeat until V_min,sys >= V_target or
   `max_correction_iters` power flows have been run.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class QResult:
    activated: bool
    iterations: int
    q_kvar: np.ndarray          # injected Q per aggregator
    shed_frac: np.ndarray       # fraction of EV power shed per aggregator
    exhausted: bool
    vmin_pre: float
    vmin_post: float
    res: object                 # final PFResult


class ReactiveController:
    def __init__(self, cfg: dict, s_rated_kva: float):
        v, rp = cfg["voltage"], cfg["reactive_power"]
        self.v_min = v["v_min"]
        self.v_target = v["v_min"] + v["correction_margin"]
        self.v_crit = v["v_critical"]
        self.max_iters = v["max_correction_iters"]
        self.fallback = rp["curtailment_fallback"]
        self.curt_step = rp["curtailment_step"]
        self.s_rated = s_rated_kva

    def capacity(self, p_kw: np.ndarray) -> float:
        return float(np.sqrt(np.maximum(self.s_rated ** 2 - p_kw ** 2, 0.0)).sum())

    def correct(self, pf, p_bus: np.ndarray, q_bus: np.ndarray, agg_bus: np.ndarray,
                ev_p_kw: list, res) -> QResult:
        n = len(agg_bus)
        q = np.zeros(n)
        shed = np.zeros(n)
        vmin_pre = res.vmin
        if vmin_pre >= self.v_target:
            return QResult(False, 0, q, shed, False, vmin_pre, vmin_pre, res)
        p_agg = np.array([float(np.sum(p)) for p in ev_p_kw])
        exhausted = False
        prev_q = prev_v = None
        iters = 0
        for _ in range(self.max_iters):
            vm = res.vm
            vsys = float(vm.min())
            if vsys >= self.v_target:
                break
            qt = float(q.sum())
            sens = None
            if prev_q is not None and qt > prev_q + 1e-6 and vsys > prev_v + 1e-7:
                sens = (vsys - prev_v) / (qt - prev_q)
            prev_q, prev_v = qt, vsys
            cap = np.array([self.capacity(ev_p_kw[k] * (1.0 - shed[k])) for k in range(n)])
            avail = cap - q
            w = np.maximum(1e-3, self.v_target - vm[agg_bus])
            active = avail > 1e-6
            progress = False
            if active.any():
                wa = np.where(active, w, 0.0)
                if sens is not None and sens > 1e-9:
                    need = 1.2 * (self.v_target - vsys) / sens * wa / wa.sum()
                else:
                    frac = np.clip((self.v_target - vsys) / (self.v_target - self.v_crit), 0, 1)
                    need = frac * avail * wa / wa.sum() * active.sum()
                need = np.minimum(np.where(active, need, 0.0), np.maximum(avail, 0.0))
                if need.max() > 1e-6:
                    q += need
                    progress = True
            if not progress:
                exhausted = True
                if not self.fallback or np.all((shed >= 1.0) | (p_agg <= 0)):
                    break
                inc = self.curt_step * w / w.max()
                shed = np.where(p_agg > 0, np.minimum(1.0, shed + inc), shed)
            p_mod = p_bus.copy()
            q_mod = q_bus.copy()
            np.add.at(p_mod, agg_bus, -p_agg * shed / 1000.0)
            np.add.at(q_mod, agg_bus, -q / 1000.0)
            res = pf.solve(p_mod, q_mod)
            iters += 1
        return QResult(True, iters, q, shed, exhausted, vmin_pre, res.vmin, res)
