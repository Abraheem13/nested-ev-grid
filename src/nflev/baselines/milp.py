"""Model-predictive LP-OPF baseline (strongest model-based comparator).

At every dispatch interval it solves, over the rest of the horizon,

  min  sum_{i,s} lambda_s c_{i,s} D + M sum_i slack_i
  s.t. sum_s eta c_{i,s} D + slack_i >= need_i                    (energy)
       0 <= c_{i,s} <= p_max a_{i,s}                               (charger, availability)
       sum_{i in k} c_{i,s} <= P_cap                               (aggregator transformer)
       V0_n(s) + sum_k S_{n,k}(s) P_k(s) >= V_min + delta  for all buses n (linearised AC voltage)

with perfect foresight of prices and of every vehicle's arrival, departure and
need, and the *forecast* (nominal) base load. V0 and S are the no-EV voltages
and finite-difference sensitivities of the full AC power flow at the forecast
operating point of each interval. Only the first interval of the plan is
applied; the problem is re-solved at the next interval with updated states
(declined, curtailed or forecast-error effects are therefore corrected).
Solved with HiGHS (scipy.optimize.linprog).
"""
from __future__ import annotations

import numpy as np
from scipy import sparse
from scipy.optimize import linprog

from .simple import Policy, _flat_corridor

SLACK_PENALTY = 5.0          # EUR per kWh of undelivered energy (>> any price)


class LPOPF(Policy):
    name = "lp_opf"

    def reset(self, env) -> None:
        self.n_tau = int(round(env.spec.horizon_h / env.interval_h))
        self.v_floor = env.v_min + env.cfg["voltage"]["correction_margin"]
        self.v0 = np.zeros((self.n_tau, env.net.n_bus))
        self.sens = np.zeros((self.n_tau, env.net.n_bus, env.n_agg))
        for s in range(self.n_tau):
            m = env.spec.load_forecast[min(s, len(env.spec.load_forecast) - 1)]
            self.v0[s], self.sens[s] = env.pf.voltage_sensitivity(
                env.p_nom * m, env.q_nom * m, env.agg_bus)
        self.solves = 0
        self.slack_kwh = 0.0

    def plan(self, env) -> dict[int, float]:
        d = env.interval_h
        tau = int(round(env.t_h / d))
        need = env.need()
        cand = np.flatnonzero((~env.departed) & (need > 1e-6) & (env.dep > env.t_h))
        if len(cand) == 0:
            return {}
        horizon = np.arange(tau, self.n_tau)
        cols, ub, ev_of, s_of = [], [], [], []
        for i in cand:
            for s in horizon:
                start, end = s * d, (s + 1) * d
                if env.arr[i] > start + 1e-9 or env.dep[i] <= start + 1e-9:
                    continue
                if s == tau and not env.connected[i]:
                    continue
                frac = min(1.0, (env.dep[i] - start) / d)
                cols.append(len(cols)); ub.append(env.p_max * frac)
                ev_of.append(i); s_of.append(s)
        nv = len(cols)
        if nv == 0:
            return {}
        ev_of, s_of = np.array(ev_of), np.array(s_of)
        ev_pos = {int(i): j for j, i in enumerate(cand)}
        ne = len(cand)
        lmp = np.array([env.lmp(s * d) for s in s_of]) / 1000.0
        c = np.concatenate([lmp * d, np.full(ne, SLACK_PENALTY)])
        rows, cidx, vals, b = [], [], [], []
        r = 0
        # energy: -eta d sum_s c - slack <= -need
        for j, i in enumerate(cand):
            m = np.flatnonzero(ev_of == i)
            rows += [r] * (len(m) + 1)
            cidx += list(m) + [nv + j]
            vals += [-env.eff * d] * len(m) + [-1.0]
            b.append(-need[i]); r += 1
        agg_of = env.agg[ev_of]
        for s in horizon:
            at_s = np.flatnonzero(s_of == s)
            if len(at_s) == 0:
                continue
            for k in range(env.n_agg):                       # transformer caps
                m = at_s[agg_of[at_s] == k]
                if len(m):
                    rows += [r] * len(m); cidx += list(m); vals += [1.0] * len(m)
                    b.append(env.p_cap); r += 1
            sk = self.sens[s] / 1000.0                       # p.u. per kW
            for n in range(1, env.net.n_bus):                 # linearised voltage floor
                coef = -sk[n, agg_of[at_s]]
                rows += [r] * len(at_s); cidx += list(at_s); vals += list(coef)
                b.append(self.v0[s, n] - self.v_floor); r += 1
        a = sparse.csr_matrix((vals, (rows, cidx)), shape=(r, nv + ne))
        bounds = [(0.0, u) for u in ub] + [(0.0, None)] * ne
        res = linprog(c, A_ub=a, b_ub=np.array(b), bounds=bounds, method="highs")
        self.solves += 1
        if not res.success:
            # c = 0, slack = need is always feasible because the calibrated no-EV
            # voltage stays above the floor; a failure is a modelling error.
            raise RuntimeError(f"LP-OPF failed: {res.message}")
        x = res.x
        self.slack_kwh = float(x[nv:].sum())
        now = np.flatnonzero(s_of == tau)
        return {int(ev_of[j]): float(x[j]) for j in now if x[j] > 1e-6}

    def act(self, env):
        _flat_corridor(env)
        plan = self.plan(env)
        for k in range(env.n_agg):
            env.set_rates(k, {i: p for i, p in plan.items() if env.agg[i] == k}, 0.0)
