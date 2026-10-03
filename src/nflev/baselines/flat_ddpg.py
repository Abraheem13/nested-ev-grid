"""Flat single-agent DDPG baseline.

One agent, one timescale (15 min). State: Level-1 features plus every
aggregator's local features (11 + 5 x 27 = 146). Action: (u_k, phi_k) for every
aggregator (10), executed through the same least-laxity-first allocation and
deadline guard as the nested framework. No pricing layer (the corridor is the
full retail range) and no Level 3: voltage enters only through the reward
penalty, so safety must be learned.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from ..agents.ddpg import DDPGAgent
from ..env.charging_env import L1_OBS_DIM, L2_OBS_DIM, ChargingEnv
from ..env.episode import make_episode
from ..training.common import (Curriculum, TrainLog, TrainState, aggregator_rewards, noise_schedule,
                               voltage_penalty)


def flat_obs(env) -> np.ndarray:
    return np.concatenate([env.l1_obs()] + [env.l2_obs(k) for k in range(env.n_agg)])


def flat_obs_dim(n_agg: int) -> int:
    return L1_OBS_DIM + n_agg * L2_OBS_DIM


def apply_flat_action(env, a: np.ndarray) -> None:
    env.set_corridor(env.price_floor, env.price_ceil)
    for k in range(env.n_agg):
        env.set_aggregate(k, a[2 * k], a[2 * k + 1])


def applied_flat_action(info: dict, a: np.ndarray) -> np.ndarray:
    out = np.array(a, np.float32).copy()
    out[0::2] = info["applied_u"]
    return out


class FlatDDPG:
    name = "flat_ddpg"

    def __init__(self, cfg: dict, n_agg: int, seed: int):
        self.agent = DDPGAgent(cfg, flat_obs_dim(n_agg), 2 * n_agg, seed)
        self.explore = False

    def reset(self, env) -> None:
        pass

    def act(self, env) -> None:
        apply_flat_action(env, self.agent.act(flat_obs(env), self.explore))

    def state_dict(self):
        return self.agent.state_dict()

    def load_state_dict(self, sd):
        self.agent.load_state_dict(sd)


def train_flat_ddpg(cfg: dict, fleet: str, network: str, seed: int, out_dir: pathlib.Path,
                    episodes: int | None = None, log_every: int = 25) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = episodes or cfg["training"]["episodes"]
    env = ChargingEnv(cfg, network, q_control=False)
    ctl = FlatDDPG(cfg, env.n_agg, seed)
    cur = Curriculum(cfg)
    out_dir.mkdir(parents=True, exist_ok=True)
    state, start = TrainState(out_dir, int(cfg["training"].get("checkpoint_every", 50))), 0
    st = state.load()
    if st is not None:                                   # resume an interrupted run
        ctl, cur, rng, start = st["ctl"], st["cur"], st["rng"], st["ep"]
        print(f"resuming {out_dir.name} at episode {start}", flush=True)
    log = TrainLog(out_dir / "train_log.csv", start)
    t_start = time.time()
    for ep in range(start, episodes):
        ctl.agent.noise = noise_schedule(cfg, ep)
        spec = make_episode(cfg, fleet, "train", cur.n_ev(rng), seed=10_000 * (seed + 1) + ep)
        env.reset(spec)
        pending = None
        t0 = time.time()
        while not env.done:
            s = flat_obs(env)
            if pending is not None:
                ctl.agent.store(*pending, s, False)
                ctl.agent.update()
            a = ctl.agent.act(s, True)
            apply_flat_action(env, a)
            info = env.run_interval()
            r = float(aggregator_rewards(cfg, env, info).sum() + voltage_penalty(env, info))
            pending = (s, applied_flat_action(info, a), r)
        ctl.agent.store(*pending, flat_obs(env), True)
        ctl.agent.update()
        m = env.episode_metrics()
        cur.report(m)
        row = {"episode": ep, "stage": cur.idx, "n_ev": env.n_ev, **m, "wall_s": round(time.time() - t0, 3)}
        log.write(row)
        state.maybe_save(ep + 1, episodes, {"ctl": ctl, "cur": cur, "rng": rng})
        if ep % log_every == 0:
            print(f"[flat_ddpg/{fleet}/{network}/s{seed}] ep{ep:4d} stage{cur.idx} cost={m['cost_eur']:.1f} "
                  f"SQ={m['service_quality']:.3f} viol={m['violation_rate_pct']:.2f}%", flush=True)
    log.close()
    torch.save({"model": ctl.state_dict(), "seed": seed, "fleet": fleet, "network": network,
                "train_wall_s": time.time() - t_start}, out_dir / "model.pt")
    state.finish()
    return {"wall_s": time.time() - t_start}
