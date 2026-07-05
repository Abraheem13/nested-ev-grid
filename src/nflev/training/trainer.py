"""Nested training loop + curriculum manager.

Reward structure (v2):
  L1 (hourly): -a1*cost -a2*vdeficit^2 -a3*1[viol] +a4*SQproxy
               -a_thermal*overload -a_curt*curtailed_kWh  (R1.7: since Level 3
               keeps recorded violations at ~0, curtailment/Q-exhaustion carry
               the safety-pressure signal the DSO must learn to relieve)
  L2 (15-min, per aggregator): -b1*price*energy -b2*unmet^2(departures)
               -b_curt*local curtailment
Curriculum advancement (R1.7): cost threshold AND Q-activation frequency AND
curtailment frequency sustained over a window — violation rate is NOT used.
"""
from __future__ import annotations
import time
import numpy as np

from ..agents.ppo import PPOAgent
from ..agents.ddpg import DDPGAgent


class Curriculum:
    def __init__(self, cfg: dict):
        self.stages = cfg["curriculum"]["stages"]
        self.window = cfg["curriculum"]["advance_window_episodes"]
        self.idx = 0
        self.history: list[dict] = []

    @property
    def stage(self) -> dict:
        return self.stages[self.idx]

    def report(self, metrics: dict) -> bool:
        """Returns True if advanced."""
        self.history.append(metrics)
        if len(self.history) < self.window or self.idx >= len(self.stages) - 1:
            return False
        w = self.history[-self.window:]
        cost_ok = np.mean([m["daily_cost_usd"] for m in w]) < 400 * self.stage["penetration"] + 100
        qact_ok = np.mean([m["q_activation_freq"] for m in w]) < 0.30
        curt_ok = np.mean([m["curtailed_kwh"] for m in w]) < 5.0
        sq_ok = np.mean([m["service_quality"] for m in w]) > 0.85
        if cost_ok and qact_ok and curt_ok and sq_ok:
            self.idx += 1
            self.history.clear()
            return True
        return False


class NestedTrainer:
    def __init__(self, cfg: dict, env_factory, device: str = "cpu",
                 max_evs: int = 40, ablation: str = "none"):
        self.cfg = cfg
        self.ablation = ablation  # none | no_l1 | flat_timescale
        self.env_factory = env_factory  # (penetration, stochastic, episode_h, seed) -> env
        self.max_evs = max_evs
        self.l1 = PPOAgent(cfg, state_dim=cfg["level1"]["state_dim"], device=device)
        n_agg = cfg["level2"]["n_aggregators"]
        l2_state_dim = 6 + max_evs * 3
        self.l2 = [DDPGAgent(cfg, l2_state_dim, max_evs, device) for _ in range(n_agg)]
        self.curriculum = Curriculum(cfg)
        self.r1 = cfg["level1"]["reward"]
        self.r2 = cfg["level2"]["reward"]

    # ------------------------------------------------------------- rewards
    def _l1_reward(self, hour_metrics: list[dict], sq: float) -> float:
        cost = sum(m["cost"] for m in hour_metrics)
        vmin = min(m["v_min"] for m in hour_metrics)
        vdef = max(0.0, 0.95 - vmin)
        viol = float(vmin < 0.95 - 1e-9)
        curt = sum(m["curtailed_kwh"] for m in hour_metrics)
        r = (-self.r1["alpha_cost"] * cost / 50.0
             - self.r1["alpha_vsq"] * (vdef * 20) ** 2
             - self.r1["alpha_vind"] * viol
             + self.r1["alpha_sat"] * sq
             - self.r1["alpha_thermal"] * curt / 10.0)
        return float(r)

    def _l2_reward(self, env, k: int, interval: dict, energy_cost: float,
                   unmet_sq: float, t_h: float) -> float:
        # v2 economics: the aggregator is a PROFIT-SEEKING retailer buying at
        # wholesale LMP and selling within the DSO corridor. The v1 reward
        # (-beta1 * p * sum(c), Eq. 9) treated revenue as cost, making
        # under-delivery locally optimal — the root cause of the SQ collapse
        # in runs s0/v2_s0. Margin-seeking also endogenously shifts charging
        # into low-LMP hours, aligning aggregator and system objectives.
        delivered_kwh = sum(env.agg_rates[k].values()) * 0.25
        margin = (env.exec_prices[k] - env._lmp(t_h) / 1000.0) * delivered_kwh
        urgency_pen = 0.0
        for ev in env.fleet.connected_by_aggregator(k):
            need_frac = max(0.0, ev.soc_target - ev.soc)
            hrs_left = max(0.25, ev.departure_h - t_h)
            urgency_pen += (need_frac ** 2) / hrs_left
        return float(2.0 * margin
                     - 4.0 * urgency_pen
                     - self.r2["beta_unmet"] * 5.0 * unmet_sq
                     - 0.5 * interval["curtailed_kwh"])

    # ------------------------------------------------------------ episode
    def run_episode(self, seed: int, train: bool = True,
                    scenario: dict | None = None,
                    price_profile=None, load_profile=None) -> dict:
        st = self.curriculum.stage if scenario is None else scenario
        env = self.env_factory(st, seed)
        s1 = env.reset(price_profile, load_profile)
        n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
        per_hour = 1 if self.ablation == "flat_timescale" else \
            int(env.pricing_s // env.dispatch_s)

        l1_s, l1_a, l1_logp = None, None, None
        hour_buf: list[dict] = []
        l2_prev = {k: None for k in range(env.n_agg)}  # (s, a)
        dep_seen: set[int] = set()

        for i in range(n_intervals):
            # ---- Level 1 acts hourly
            if i % per_hour == 0 and self.ablation == "no_l1":
                env.set_corridor(0.10, 0.14)
            elif i % per_hour == 0:
                if l1_s is not None and train:
                    r1 = self._l1_reward(hour_buf, env.fleet.service_quality())
                    self.l1.store(l1_s, l1_a, l1_logp, r1, False)
                s1 = env._l1_state()
                a1, logp1 = self.l1.act(s1, deterministic=not train)
                l1_s, l1_a, l1_logp, hour_buf = s1, a1, logp1, []
                l1c = self.cfg["level1"]
                p_min = l1c["price_floor"] + a1[0] * (l1c["price_ceil"] - l1c["price_floor"] - 0.05)
                width = l1c["min_corridor_width"] + a1[1] * 0.10
                env.set_corridor(p_min, p_min + width)

            # ---- Level 2 acts per interval
            l2_states, l2_actions = {}, {}
            for k in range(env.n_agg):
                s2 = env.l2_state(k, self.max_evs)
                a2 = self.l2[k].act(s2, deterministic=not train)
                rates = a2[:self.max_evs] * self.cfg["reactive_power"]["charger_p_max_kw"]
                env.set_dispatch(k, rates, price_frac=float(a2[-1]))
                l2_states[k], l2_actions[k] = s2, a2

            interval = env.run_dispatch_interval()
            hour_buf.append(interval)

            # ---- L2 rewards + storage
            for k in range(env.n_agg):
                energy_cost = env.exec_prices[k] * sum(env.agg_rates[k].values()) * 0.25
                unmet_sq = 0.0
                for ev in env.evs:
                    if ev.departed and ev.aggregator == k and ev.idx not in dep_seen:
                        unmet_sq += (max(0.0, ev.soc_target - ev.soc)) ** 2
                        dep_seen.add(ev.idx)
                r2 = self._l2_reward(env, k, interval, energy_cost, unmet_sq,
                                     t_h=env.t_s / 3600.0)
                if l2_prev[k] is not None and train:
                    ps, pa = l2_prev[k]
                    self.l2[k].store(ps, pa, r2, l2_states[k], i == n_intervals - 1)
                    self.l2[k].update()
                l2_prev[k] = (l2_states[k], l2_actions[k])

        # close L1 trajectory + PPO update
        if train and l1_s is not None and self.ablation != "no_l1":
            r1 = self._l1_reward(hour_buf, env.fleet.service_quality())
            self.l1.store(l1_s, l1_a, l1_logp, r1, True)
            self.l1.update()

        m = env.episode_metrics()
        if train:
            advanced = self.curriculum.report(m)
            m["curriculum_stage"] = self.curriculum.idx
            m["advanced"] = advanced
        return m