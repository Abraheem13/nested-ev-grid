"""Full AC power flow for radial feeders (backward/forward sweep in Z-bus form).

For a radial network with constant-power loads the AC power-flow equations are
exactly

    V = V0 * 1 - Zpath @ conj(S / V),

where S is the complex load (minus generation) at every non-substation bus and
Zpath[i, j] is the impedance shared by the root->i and root->j paths. The
fixed-point iteration below is the vectorised backward/forward sweep; it solves
the same nonlinear equations as Newton-Raphson. `tests/test_powerflow.py` checks
it against pandapower's Newton-Raphson to 1e-8 p.u. on both feeders.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .network import LINE_AMPACITY_KA, SUBSTATION_MVA, RadialNetwork


@dataclass
class PFResult:
    vm: np.ndarray            # |V| per bus (p.u.), bus 0 = substation
    v: np.ndarray             # complex voltages (p.u.)
    i_branch_pu: np.ndarray   # complex current in the branch feeding bus i (index i-1)
    s_sub_mva: complex        # power drawn from the substation
    iterations: int

    @property
    def vmin(self) -> float:
        return float(self.vm.min())


class RadialPowerFlow:
    def __init__(self, net: RadialNetwork, v0: float = 1.03, tol: float = 1e-10,
                 max_iter: int = 100):
        self.net = net
        self.v0 = v0
        self.tol = tol
        self.max_iter = max_iter
        self.zpath = net.z_path()
        self.tmat = net.downstream()
        self._v_last = np.full(net.n_bus - 1, v0, dtype=complex)

    def solve(self, p_mw: np.ndarray, q_mvar: np.ndarray, warm: bool = True) -> PFResult:
        """p_mw, q_mvar: net load per bus (positive = consumption), length n_bus.
        Entries at bus 0 are ignored (the substation is the slack bus)."""
        s = (np.asarray(p_mw[1:]) + 1j * np.asarray(q_mvar[1:])) / self.net.base_mva
        v = self._v_last.copy() if warm else np.full(self.net.n_bus - 1, self.v0, dtype=complex)
        for it in range(1, self.max_iter + 1):
            i_bus = np.conj(s / v)
            v_new = self.v0 - self.zpath @ i_bus
            if np.max(np.abs(v_new - v)) < self.tol:
                v = v_new
                break
            v = v_new
        else:
            raise RuntimeError("radial power flow did not converge")
        self._v_last = v
        i_bus = np.conj(s / v)
        i_branch = self.tmat @ i_bus
        s_sub = self.v0 * np.conj(i_bus.sum()) * self.net.base_mva
        v_all = np.concatenate([[self.v0 + 0j], v])
        return PFResult(vm=np.abs(v_all), v=v_all, i_branch_pu=i_branch,
                        s_sub_mva=complex(s_sub), iterations=it)

    # ------------------------------------------------------------ features
    def line_margin(self, res: PFResult) -> float:
        """1 - max line loading (fraction of the assumed ampacity)."""
        i_ka = np.abs(res.i_branch_pu) * self.net.i_base_ka
        return float(1.0 - i_ka.max() / LINE_AMPACITY_KA)

    def substation_loading(self, res: PFResult) -> float:
        return float(abs(res.s_sub_mva) / SUBSTATION_MVA[self.net.name])

    # ----------------------------------------------------------- linearization
    def voltage_sensitivity(self, p_mw: np.ndarray, q_mvar: np.ndarray, buses,
                            dp_mw: float = 0.01) -> tuple[np.ndarray, np.ndarray]:
        """Finite-difference d|V|/dP (p.u. per MW) at every bus for an active-power
        increase at each bus in `buses`. Returns (vm_base, S) with S[n, k]."""
        base = self.solve(p_mw, q_mvar, warm=False).vm
        sens = np.zeros((self.net.n_bus, len(buses)))
        for k, b in enumerate(buses):
            p2 = np.array(p_mw, dtype=float).copy()
            p2[b] += dp_mw
            sens[:, k] = (self.solve(p2, q_mvar, warm=False).vm - base) / dp_mw
        self.solve(p_mw, q_mvar, warm=False)
        return base, sens
