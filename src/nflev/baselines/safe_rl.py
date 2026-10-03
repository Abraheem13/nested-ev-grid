"""Constrained-RL baselines (PPO-Lagrangian and CPO) on the flat interface.

Same state/action/allocation as flat DDPG; reward = sum of aggregator rewards;
constraint cost per interval = 1[V_min < 0.95] + 10 max(0, 0.95 - V_min) with
limit d = 0. One policy update per training episode. No Level 3.
"""
from __future__ import annotations

import copy
import pathlib
import time

import numpy as np
import torch

from ..agents.cpo import CPO
from ..agents.ppo_lagrangian import PPOLagrangian
from ..env.charging_env import ChargingEnv
from ..env.episode import make_episode
from ..training.common import Curriculum, RetailMultiplier, TrainLog, TrainState, aggregator_rewards, voltage_cost
from .flat_ddpg import apply_flat_action, flat_obs, flat_obs_dim

AGENTS = {"ppo_lag": PPOLagrangian, "cpo": CPO}


class SafeRL:
    def __init__(self, cfg: dict, n_agg: int, method: str):
        self.name = method
        c = copy.deepcopy(cfg)          # acts every 15 min: same discount as the other dispatch agents
        c["training"]["ppo"]["gamma"] = cfg["training"]["ddpg"]["gamma"]
        self.agent = AGENTS[method](c, flat_obs_dim(n_agg), 2 * n_agg)
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
    mult = RetailMultiplier(cfg)
    state, start = TrainState(out_dir, int(cfg["training"].get("checkpoint_every", 50))), 0
    st = state.load()
    if st is not None:                                   # resume an interrupted run
        ctl, cur, rng, mult, start = st["ctl"], st["cur"], st["rng"], st["mult"], st["ep"]
        print(f"resuming {out_dir.name} at episode {start}", flush=True)
    log = TrainLog(out_dir / "train_log.csv", start)
    t_start = time.time()
    for ep in range(start, episodes):
        spec = make_episode(cfg, fleet, "train", cur.n_ev(rng), seed=10_000 * (seed + 1) + ep)
        env.reset(spec)
        t0 = time.time()
        while not env.done:
            s = flat_obs(env)
            a, raw, logp = ctl.agent.act(s, True)
            apply_flat_action(env, a)
            info = env.run_interval()
            ctl.agent.store(s, raw, logp, float(aggregator_rewards(cfg, env, info, mult.w).sum()),
                            voltage_cost(env, info))
        ctl.agent.update()
        m = env.episode_metrics()
        cur.report(m)
        row = {"episode": ep, "stage": cur.idx, "n_ev": env.n_ev, **m, "w_retail": mult.w,
               "wall_s": round(time.time() - t0, 3)}
        mult.update(m)
        log.write(row)
        state.maybe_save(ep + 1, episodes, {"ctl": ctl, "cur": cur, "rng": rng, "mult": mult})
        if ep % log_every == 0:
            print(f"[{method}/{fleet}/{network}/s{seed}] ep{ep:4d} stage{cur.idx} cost={m['cost_eur']:.1f} "
                  f"SQ={m['service_quality']:.3f} viol={m['violation_rate_pct']:.2f}%", flush=True)
    log.close()
    torch.save({"model": ctl.state_dict(), "seed": seed, "fleet": fleet, "network": network,
                "train_wall_s": time.time() - t_start}, out_dir / "model.pt")
    state.finish()
    return {"wall_s": time.time() - t_start}
