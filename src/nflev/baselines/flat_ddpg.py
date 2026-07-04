"""Flat single-agent DDPG baseline (v1 paper's principal learning baseline).

One monolithic agent observes the concatenated system state and outputs all
charging rates + prices at a single 15-minute timescale. Voltage feasibility
is handled only by a penalty term (r -= penalty per violation); the Level 3
reactive controller is DISABLED so the comparison isolates the nested
architecture + physics layer contribution, exactly as in v1 Table IV.

State: L1 features (8) + per-aggregator [head(6) + 20 EV slots x 3] = 338-d
Action: 5 x (20 rates + 1 price) = 105-d
"""
from __future__ import annotations
import numpy as np

from ..agents.ddpg import DDPGAgent

MAX_EVS_FLAT = 20  # per aggregator slots in flat state/action


class FlatDDPG:
    name = "flat_ddpg"

    def __init__(self, cfg: dict, n_agg: int = 5, device: str = "cpu",
                 violation_penalty: float = 1000.0):
        self.cfg, self.n_agg = cfg, n_agg
        self.state_dim = 8 + n_agg * (6 + MAX_EVS_FLAT * 3)
        self.pen = violation_penalty
        # reuse DDPGAgent with a custom action head size via max_evs trick
        self.agent = DDPGAgent(cfg, self.state_dim,
                               max_evs=n_agg * (MAX_EVS_FLAT + 1) - 1,
                               device=device)

    def flat_state(self, env) -> np.ndarray:
        parts = [env._l1_state()]
        for k in range(self.n_agg):
            parts.append(env.l2_state(k, MAX_EVS_FLAT))
        return np.concatenate(parts).astype(np.float32)

    def apply_action(self, env, a: np.ndarray):
        env.set_corridor(0.05, 0.30)  # flat agent has no corridor mechanism
        per = MAX_EVS_FLAT + 1
        c_max = self.cfg["reactive_power"]["charger_p_max_kw"]
        for k in range(self.n_agg):
            blk = a[k * per:(k + 1) * per]
            env.set_dispatch(k, blk[:MAX_EVS_FLAT] * c_max,
                             price_frac=float(blk[-1]))

    def reward(self, env, interval: dict, dep_unmet_sq: float) -> float:
        cost = interval["cost"]
        viol = float(interval["v_min"] < 0.95 - 1e-9)
        return float(-cost / 10.0 - self.pen / 100.0 * viol - 20.0 * dep_unmet_sq)

    # ------------------------------------------------------------ episode
    def run_episode(self, env, price_profile, load_profile,
                    train: bool = True) -> dict:
        env.reset(price_profile, load_profile)
        n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
        prev = None
        dep_seen: set[int] = set()
        for i in range(n_intervals):
            s = self.flat_state(env)
            a = self.agent.act(s, deterministic=not train)
            self.apply_action(env, a)
            interval = env.run_dispatch_interval()
            unmet = 0.0
            for ev in env.evs:
                if ev.departed and ev.idx not in dep_seen:
                    unmet += max(0.0, ev.soc_target - ev.soc) ** 2
                    dep_seen.add(ev.idx)
            r = self.reward(env, interval, unmet)
            if prev is not None and train:
                self.agent.store(prev[0], prev[1], r, s, i == n_intervals - 1)
                self.agent.update()
            prev = (s, a)
        return env.episode_metrics()