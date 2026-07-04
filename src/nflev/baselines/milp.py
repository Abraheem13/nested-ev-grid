"""Centralized optimization baseline: perfect-information day-ahead LP.

Formulation (matches the paper's 'Centralized MILP' baseline):
  min  sum_{i,tau} LMP(tau) * c_{i,tau} * dtau
  s.t. sum_tau eta * c_{i,tau} * dtau >= need_i            (energy delivery)
       0 <= c_{i,tau} <= c_max * 1[connected(i,tau)]        (charger limits)
       sum_{i in agg k} c_{i,tau} <= P_cap                  (transformer)
       vmin_base(tau) + sum_k S_k * P_k(tau) >= 0.95        (LINEARIZED voltage)

Voltage is linearized via empirically identified sensitivities S_k = dVmin/dP_k
(one PF perturbation per aggregator bus). The deliberate scientific point:
this baseline is optimal under its linearization, but when its schedule is
replayed through the FULL AC environment the linearization error surfaces —
reproducing the documented failure mode (v1 Table IV: MILP vmin 0.939).
Solved with HiGHS through scipy.linprog. Energy delivery is soft (slack with
high penalty) so the LP remains feasible at extreme penetration; binding slack
is reported as infeasibility, reproducing the S5 'Infeasible' entry honestly.
"""
from __future__ import annotations
import numpy as np
from scipy.optimize import linprog
import pandapower as pp


def identify_sensitivities(env, probe_kw: float = 200.0) -> tuple[float, np.ndarray]:
    """Empirical dVmin/dP_k (p.u. per kW) at each aggregator bus."""
    env._apply_loads(6.0)   # episode t=6 == 18:00 wall clock (peak)
    env._solved_pf()
    v0 = float(env.net.res_bus.vm_pu.min())
    sens = np.zeros(env.n_agg)
    for k in range(env.n_agg):
        li = env.ev_load_idx[k]
        p_orig = float(env.net.load.at[li, "p_mw"])
        env.net.load.at[li, "p_mw"] = p_orig + probe_kw / 1000.0
        pp.runpp(env.net, init="results", numba=True)
        sens[k] = (float(env.net.res_bus.vm_pu.min()) - v0) / probe_kw
        env.net.load.at[li, "p_mw"] = p_orig
    pp.runpp(env.net, init="results", numba=True)
    return v0, sens  # sens is negative (load depresses voltage)


def base_vmin_profile(env, load_profile) -> np.ndarray:
    """Predicted no-EV vmin per 15-min interval (perfect load foresight)."""
    n = int(env.episode_h * 4)
    out = np.zeros(n)
    for tau in range(n):
        for k in range(env.n_agg):
            env.net.load.at[env.ev_load_idx[k], "p_mw"] = 0.0
        env._apply_loads(tau / 4.0)
        env._solved_pf()
        out[tau] = float(env.net.res_bus.vm_pu.min())
    return out


class MILPPolicy:
    """Solves the day-ahead LP once at reset, then replays the schedule."""
    name = "milp"

    def __init__(self, v_floor: float = 0.95, slack_penalty: float = 1e4):
        self.v_floor = v_floor
        self.slack_penalty = slack_penalty
        self.schedule = None       # (n_ev, n_tau) kW
        self.infeasible = False

    def solve(self, env, price_profile, load_profile) -> bool:
        evs = env.evs
        n_ev = len(evs)
        n_tau = int(env.episode_h * 4)
        dtau, eta = 0.25, env.eff
        c_max = env.cfg["reactive_power"]["charger_p_max_kw"]

        conn = np.zeros((n_ev, n_tau), bool)
        for i, e in enumerate(evs):
            a, d = int(np.floor(e.arrival_h * 4)), int(np.ceil(e.departure_h * 4))
            conn[i, max(0, a):min(n_tau, d)] = True

        v0, sens = identify_sensitivities(env)
        vbase = base_vmin_profile(env, load_profile)

        nv = n_ev * n_tau            # charging rates
        ns = n_ev                    # energy slack
        idx = lambda i, t: i * n_tau + t
        lmp = np.array([price_profile[min(int(t / 4), len(price_profile) - 1)]
                        for t in range(n_tau)]) / 1000.0  # $/kWh

        c_obj = np.concatenate([np.tile(lmp * dtau, n_ev),
                                np.full(ns, self.slack_penalty)])

        A_ub, b_ub = [], []
        # transformer caps: sum_{i in k} c_{i,t} <= P_cap
        for k in range(env.n_agg):
            members = [i for i, e in enumerate(evs) if e.aggregator == k]
            for t in range(n_tau):
                row = np.zeros(nv + ns)
                for i in members:
                    row[idx(i, t)] = 1.0
                A_ub.append(row); b_ub.append(env.p_cap)
        # linearized voltage: -sum_k S_k * P_k(t) <= vbase(t) - v_floor
        for t in range(n_tau):
            row = np.zeros(nv + ns)
            for i, e in enumerate(evs):
                row[idx(i, t)] = -sens[e.aggregator]   # sens<0 -> coeff>0
            A_ub.append(row); b_ub.append(vbase[t] - self.v_floor)
        # energy: -eta*dtau*sum_t c_{i,t} - slack_i <= -need_i
        for i, e in enumerate(evs):
            row = np.zeros(nv + ns)
            row[i * n_tau:(i + 1) * n_tau] = -eta * dtau
            row[nv + i] = -1.0
            A_ub.append(row); b_ub.append(-e.initial_need_kwh)

        ub = np.where(conn.reshape(-1), c_max, 0.0)
        bounds = [(0.0, float(u)) for u in ub] + [(0.0, None)] * ns

        res = linprog(c_obj, A_ub=np.array(A_ub), b_ub=np.array(b_ub),
                      bounds=bounds, method="highs")
        if not res.success:
            self.infeasible = True
            return False
        slack = res.x[nv:]
        self.infeasible = bool(slack.sum() > 0.05 * sum(e.initial_need_kwh for e in evs))
        self.schedule = res.x[:nv].reshape(n_ev, n_tau)
        return True

    def act(self, env):
        tau = int(env.t_s // env.dispatch_s)
        env.set_corridor(0.10, 0.14)
        for k in range(env.n_agg):
            evs = env.fleet.connected_by_aggregator(k)
            rates = np.array([self.schedule[e.idx, tau] if self.schedule is not None
                              else 0.0 for e in evs] or [0.0])
            env.set_dispatch(k, rates, price_frac=0.5)


def run_milp_episode(env, price_profile, load_profile) -> dict:
    env.reset(price_profile, load_profile)
    pol = MILPPolicy()
    ok = pol.solve(env, price_profile, load_profile)
    # re-reset: solve() perturbed the network for identification
    env.reset(price_profile, load_profile)
    if not ok:
        m = {k: float("nan") for k in ["daily_cost_usd", "min_voltage_pu",
             "violation_rate_pct", "service_quality", "q_activation_freq",
             "curtailed_kwh", "peak_mw"]}
        m["infeasible"] = True
        return m
    n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
    for _ in range(n_intervals):
        pol.act(env)
        env.run_dispatch_interval()
    m = env.episode_metrics()
    m["infeasible"] = pol.infeasible
    return m