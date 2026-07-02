"""Physics-based reactive power controller (Level 3, fast sub-layer).

Key v2 changes addressing reviewer concerns:

R1.2 / R2.2 (deadlock): Q capacity is derived from the inverter APPARENT
power rating, not the instantaneous charging power:

    Qmax_i = sqrt(S_rated_i^2 - P_i^2)

so a charger at P=0 provides FULL reactive capacity (STATCOM-mode operation,
experimentally validated for EV chargers by Rafi et al., IEEE TVT 2022).
The v1 formulation Qmax = P_ch * tan(acos(0.95)) collapsed to zero when
charging was curtailed; that coupling is removed.

R2.2 / R2.3 (guarantee conditions): the controller targets v_min + margin,
iterates the correction loop to convergence, and when reactive capacity is
exhausted falls back to active-power curtailment (last resort). The
sufficient condition for zero recorded violations is therefore explicit and
testable: adequate aggregate S_rated at violated buses OR curtailable load.

R1.1 / R3.3 (framing): this module is documented and reported as fast
corrective control with sub-timestep convergence, not a priori constraint
enforcement.
"""
from __future__ import annotations
from dataclasses import dataclass, field
import numpy as np
import pandapower as pp


@dataclass
class QControlResult:
    activated: bool = False
    iterations: int = 0
    q_injected_kvar: dict = field(default_factory=dict)   # bus -> kVAR
    curtailed_kw: dict = field(default_factory=dict)      # bus -> kW shed
    q_exhausted: bool = False
    v_min_pre: float = 1.0
    v_min_post: float = 1.0


class ReactivePowerController:
    """Proportional Q injection with correction-loop convergence and
    curtailment fallback. Operates within one QSTS timestep."""

    def __init__(self, v_min: float, v_critical: float, margin: float,
                 max_iters: int, curtailment_fallback: bool, curtailment_step: float):
        self.v_min = v_min
        self.v_crit = v_critical
        self.v_target = v_min + margin
        self.max_iters = max_iters
        self.fallback = curtailment_fallback
        self.curt_step = curtailment_step

    @staticmethod
    def q_capacity_kvar(s_rated_kva: np.ndarray, p_kw: np.ndarray) -> np.ndarray:
        """Qmax_i = sqrt(S^2 - P^2); inverter-rating based (deadlock fix)."""
        return np.sqrt(np.maximum(s_rated_kva**2 - p_kw**2, 0.0))

    def proportional_q(self, v_local: float, q_max: float) -> float:
        """Eq. (11) v2: proportional in deviation below v_target, saturating
        at full capacity at v_critical."""
        if v_local >= self.v_target:
            return 0.0
        frac = (self.v_target - v_local) / (self.v_target - self.v_crit)
        return float(q_max * np.clip(frac, 0.0, 1.0))

    def correct(self, net: pp.pandapowerNet, charger_map: dict) -> QControlResult:
        """Run the within-timestep correction loop.

        charger_map: bus_idx -> dict(sgen_idx, s_rated_kva (array), p_kw (array),
                                     load_idx for curtailment)
        Assumes net has been solved (runpp) prior to call.
        """
        res = QControlResult(v_min_pre=float(net.res_bus.vm_pu.min()))
        if res.v_min_pre >= self.v_min:
            res.v_min_post = res.v_min_pre
            return res

        res.activated = True
        q_prev, v_prev = 0.0, res.v_min_pre
        for it in range(self.max_iters):
            res.iterations = it + 1
            vm = net.res_bus.vm_pu
            v_sys_min = float(vm.min())
            if v_sys_min >= self.v_target:
                break
            # Sensitivity-based closure (Newton step on measured dV/dQ) once a
            # first proportional injection has established the local slope.
            q_total = sum(res.q_injected_kvar.values())
            sens = None
            if q_total > q_prev + 1e-6 and v_sys_min > v_prev + 1e-7:
                sens = (v_sys_min - v_prev) / (q_total - q_prev)  # p.u. per kVAR
            q_prev, v_prev = q_total, v_sys_min
            # System-driven: a violation anywhere mobilizes all charger buses,
            # weighted by local depression (voltage-sensitivity proxy).
            progress = False
            weights = {}
            for b, cm in charger_map.items():
                q_max_each = self.q_capacity_kvar(cm["s_rated_kva"], cm["p_kw"])
                q_avail = float(q_max_each.sum()) - res.q_injected_kvar.get(b, 0.0)
                if q_avail <= 1e-6:
                    continue
                w = max(1e-3, self.v_target - float(vm.at[b]))
                weights[b] = (w, q_avail)
            if weights:
                w_sum = sum(w for w, _ in weights.values())
                q_req_total = None
                if sens is not None and sens > 1e-9:
                    # Newton step with 20% overshoot margin, targeting v_target
                    q_req_total = 1.2 * (self.v_target - v_sys_min) / sens
                for b, (w, q_avail) in weights.items():
                    if q_req_total is not None:
                        q_need = q_req_total * (w / w_sum)
                    else:
                        q_drive = self.proportional_q(v_sys_min, q_avail)
                        q_need = q_drive * (w / w_sum) * len(weights)
                    q_need = min(q_need, q_avail)
                    if q_need <= 1e-6:
                        continue
                    self._inject(net, charger_map[b]["sgen_idx"], q_need)
                    res.q_injected_kvar[b] = res.q_injected_kvar.get(b, 0.0) + q_need
                    progress = True

            if not progress:
                res.q_exhausted = True
                if not self.fallback:
                    break
                if not self._curtail(net, charger_map, vm, res):
                    break  # nothing left to shed
            pp.runpp(net, init="results", numba=True)

        res.v_min_post = float(net.res_bus.vm_pu.min())
        return res

    @staticmethod
    def _inject(net: pp.pandapowerNet, sgen_idx: int, q_kvar: float) -> None:
        net.sgen.at[sgen_idx, "q_mvar"] += q_kvar / 1000.0

    def _curtail(self, net, charger_map, vm, res: QControlResult) -> bool:
        """Shed a fraction of EV charging power at violated buses (last resort)."""
        shed_any = False
        if float(vm.min()) >= self.v_target:
            return False
        for b, cm in charger_map.items():
            load_idx = cm["load_idx"]
            p_now = float(net.load.at[load_idx, "p_mw"]) * 1000.0
            shed = p_now * self.curt_step
            if shed <= 1e-6:
                continue
            net.load.at[load_idx, "p_mw"] = (p_now - shed) / 1000.0
            cm["p_kw"] = cm["p_kw"] * (1 - self.curt_step)
            res.curtailed_kw[b] = res.curtailed_kw.get(b, 0.0) + shed
            shed_any = True
        return shed_any
