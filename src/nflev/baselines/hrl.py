"""Learned hierarchical RL baseline (goal-conditioned, HIRO-style).

Answers R2.1/R3.2 ("how does nested learning differ from hierarchical RL?")
with a direct empirical comparator: a two-level LEARNED hierarchy in which

  High level (hourly, DDPG):  observes the 8-d system state, outputs a power
      budget fraction per aggregator (5-d goal in [0,1]).
  Low level (15-min, DDPG per aggregator): observes l2 state + its budget
      goal, outputs rates + price; intrinsic reward = task reward minus a
      goal-tracking penalty |sum(rates) - budget|.

Distinctions vs the proposed nested framework, made measurable here:
  (i) both levels are parametric/learned — no non-parametric physics level;
 (ii) inter-level signal is a learned goal, not a physical constraint;
(iii) voltage handled by penalty only (Q-control disabled).
"""
from __future__ import annotations
import numpy as np

from ..agents.ddpg import DDPGAgent


class HRLBaseline:
    name = "hrl"

    def __init__(self, cfg: dict, n_agg: int = 5, max_evs: int = 40,
                 device: str = "cpu", violation_penalty: float = 1000.0):
        self.cfg, self.n_agg, self.max_evs = cfg, n_agg, max_evs
        self.pen = violation_penalty
        # high level: state 8 -> action 5 (budgets); reuse DDPG with max_evs=4
        self.high = DDPGAgent(cfg, state_dim=8, max_evs=n_agg - 1, device=device)
        lo_dim = 6 + max_evs * 3 + 1  # l2 state + goal
        self.low = [DDPGAgent(cfg, lo_dim, max_evs, device) for _ in range(n_agg)]

    def run_episode(self, env, price_profile, load_profile,
                    train: bool = True) -> dict:
        env.reset(price_profile, load_profile)
        n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
        per_hour = int(env.pricing_s // env.dispatch_s)
        c_max = self.cfg["reactive_power"]["charger_p_max_kw"]

        hi_prev, hi_reward_acc = None, 0.0
        lo_prev = {k: None for k in range(self.n_agg)}
        goals = np.full(self.n_agg, 0.5)
        dep_seen: set[int] = set()

        for i in range(n_intervals):
            if i % per_hour == 0:
                s_hi = env._l1_state()
                if hi_prev is not None and train:
                    self.high.store(hi_prev[0], hi_prev[1], hi_reward_acc,
                                    s_hi, False)
                    self.high.update()
                goals = self.high.act(s_hi, deterministic=not train)
                hi_prev, hi_reward_acc = (s_hi, goals.copy()), 0.0

            env.set_corridor(0.05, 0.30)
            lo_states, lo_actions = {}, {}
            for k in range(self.n_agg):
                s_lo = np.concatenate([env.l2_state(k, self.max_evs),
                                       [goals[k]]]).astype(np.float32)
                a = self.low[k].act(s_lo, deterministic=not train)
                env.set_dispatch(k, a[:self.max_evs] * c_max,
                                 price_frac=float(a[-1]))
                lo_states[k], lo_actions[k] = s_lo, a

            interval = env.run_dispatch_interval()
            viol = float(interval["v_min"] < 0.95 - 1e-9)
            hi_reward_acc += -interval["cost"] / 10.0 - self.pen / 100.0 * viol \
                             + 2.0 * env.fleet.service_quality()

            for k in range(self.n_agg):
                budget_kw = goals[k] * env.p_cap
                actual_kw = sum(env.agg_rates[k].values())
                unmet = 0.0
                for ev in env.evs:
                    if ev.departed and ev.aggregator == k and ev.idx not in dep_seen:
                        unmet += max(0.0, ev.soc_target - ev.soc) ** 2
                        dep_seen.add(ev.idx)
                lmp = env._lmp(env.t_s / 3600.0) / 1000.0
                margin = (env.exec_prices[k] - lmp) * actual_kw * 0.25
                r_lo = (2.0 * margin - abs(actual_kw - budget_kw) / env.p_cap
                        - 100.0 * unmet - 2.0 * viol)
                if lo_prev[k] is not None and train:
                    ps, pa = lo_prev[k]
                    self.low[k].store(ps, pa, r_lo, lo_states[k],
                                      i == n_intervals - 1)
                    self.low[k].update()
                lo_prev[k] = (lo_states[k], lo_actions[k])
        return env.episode_metrics()

    def save(self, prefix):
        self.high.save(f"{prefix}_high.pt")
        for k, ag in enumerate(self.low):
            ag.save(f"{prefix}_low{k}.pt")

    def load(self, prefix):
        self.high.load(f"{prefix}_high.pt")
        for k, ag in enumerate(self.low):
            ag.load(f"{prefix}_low{k}.pt")