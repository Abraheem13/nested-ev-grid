"""Constrained-RL baselines (PPO-Lagrangian and CPO) on the flat interface.

Same state/action/allocation as flat DDPG; reward = sum of aggregator rewards;
constraint cost per interval = 1[V_min < 0.95] + 10 max(0, 0.95 - V_min) with
limit d = 0. One policy update per training episode. No Level 3.
"""
from __future__ import annotations

import csv
import pathlib
import time

import numpy as np
import torch

from ..agents.cpo import CPO
from ..agents.ppo_lagrangian import PPOLagrangian
from ..env.charging_env import ChargingEnv
from ..env.episode import make_episode
from ..training.common import Curriculum, aggregator_rewards, voltage_cost
from .flat_ddpg import apply_flat_action, flat_obs, flat_obs_dim

AGENTS = {"ppo_lag": PPOLagrangian, "cpo": CPO}


class SafeRL:
    def __init__(self, cfg: dict, n_agg: int, method: str):
        self.name = method
        self.agent = AGENTS[method](cfg, flat_obs_dim(n_agg), 2 * n_agg)
        self.explore = False

    def reset(self, env) -> None:
        pass

    def act(self, env) -> None:
        a, _, _ = self.agent.act(flat_obs(env), self.explore)
        apply_flat_action(env, a)

    def state_dict(self):
        return self.agent.state_dict()

    def load_state_dict(self, sd):
        self.agent.load_state_dict(sd)


def train_safe_rl(cfg: dict, method: str, fleet: str, network: str, seed: int,
                  out_dir: pathlib.Path, episodes: int | None = None, log_every: int = 25) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = episodes or cfg["training"]["episodes"]
    env = ChargingEnv(cfg, network, q_control=False)
    ctl = SafeRL(cfg, env.n_agg, method)
    cur = Curriculum(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    f = open(out_dir / "train_log.csv", "w", newline="")
    w = None
    t_start = time.time()
    for ep in range(episodes):
        spec = make_episode(cfg, fleet, "train", cur.n_ev(rng), seed=10_000 * (seed + 1) + ep)
        env.reset(spec)
        t0 = time.time()
        while not env.done:
            s = flat_obs(env)
            a, raw, logp = ctl.agent.act(s, True)
            apply_flat_action(env, a)
            info = env.run_interval()
            ctl.agent.store(s, raw, logp, float(aggregator_rewards(cfg, env, info).sum()),
                            voltage_cost(env, info))
        ctl.agent.update()
        m = env.episode_metrics()
        cur.report(m)
        row = {"episode": ep, "stage": cur.idx, "n_ev": env.n_ev, **m, "wall_s": round(time.time() - t0, 3)}
        if w is None:
            w = csv.DictWriter(f, fieldnames=list(row))
            w.writeheader()
        w.writerow(row)
        f.flush()
        if ep % log_every == 0:
            print(f"[{method}/{fleet}/{network}/s{seed}] ep{ep:4d} stage{cur.idx} cost={m['cost_eur']:.1f} "
                  f"SQ={m['service_quality']:.3f} viol={m['violation_rate_pct']:.2f}%", flush=True)
    f.close()
    torch.save({"model": ctl.state_dict(), "seed": seed, "fleet": fleet, "network": network,
                "train_wall_s": time.time() - t_start}, out_dir / "model.pt")
    return {"wall_s": time.time() - t_start}
