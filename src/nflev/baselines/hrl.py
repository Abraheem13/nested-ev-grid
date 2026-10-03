"""Goal-conditioned hierarchical RL baseline (two learned levels, no Level 3).

High level (DDPG, hourly): Level-1 features -> a power-budget fraction g_k in
[0, 1] for every aggregator. Low level (one DDPG per aggregator, 15 min): local
features plus g_k -> (u_k, phi_k), executed through the same allocation layer.
Low-level reward = aggregator reward + voltage penalty - 5 |u_k - g_k|;
high-level reward = sum over the hour of (sum of aggregator rewards + voltage
penalty). Differences from the nested framework: both levels are learned, the
inter-level signal is a learned goal instead of a price corridor, and there is
no physics level, so safety must be learned.
"""
from __future__ import annotations

import pathlib
import time

import numpy as np
import torch

from ..agents.ddpg import DDPGAgent
from ..env.charging_env import L1_OBS_DIM, L2_OBS_DIM, ChargingEnv
from ..env.episode import make_episode
from ..training.common import (Curriculum, RetailMultiplier, TrainLog, TrainState, aggregator_rewards,
                               neutral_corridor, noise_schedule, voltage_penalty)

GOAL_WEIGHT = 5.0


class HRL:
    name = "hrl"

    def __init__(self, cfg: dict, n_agg: int, seed: int):
        self.high = DDPGAgent(cfg, L1_OBS_DIM, n_agg, seed * 100 + 99)
        self.shared = bool(cfg["level2"].get("shared", False))   # same option as the nested L2
        self.n_agg = n_agg
        if self.shared:
            agent = DDPGAgent(cfg, L2_OBS_DIM + 1 + n_agg, 2, seed * 100)
            self.low = [agent] * n_agg
        else:
            self.low = [DDPGAgent(cfg, L2_OBS_DIM + 1, 2, seed * 100 + k) for k in range(n_agg)]
        self.explore = False
        self.period = 4

    def reset(self, env) -> None:
        self.interval = 0
        self.goals = np.full(env.n_agg, 0.5, np.float32)
        self.period = int(env.cfg["simulation"]["pricing_interval_s"] // env.cfg["simulation"]["dispatch_interval_s"])

    def low_obs(self, env, k):
        parts = [env.l2_obs(k), [self.goals[k]]]
        if self.shared:
            parts.append(np.eye(self.n_agg)[k])
        return np.concatenate(parts).astype(np.float32)

    def act(self, env) -> None:
        if self.interval % self.period == 0:
            self.goals = self.high.act(env.l1_obs(), self.explore)
        env.set_corridor(*neutral_corridor(env))
        for k in range(env.n_agg):
            a = self.low[k].act(self.low_obs(env, k), self.explore)
            env.set_aggregate(k, a[0], a[1])
        self.interval += 1

    def state_dict(self):
        low = [self.low[0].state_dict()] if self.shared else [a.state_dict() for a in self.low]
        return {"high": self.high.state_dict(), "low": low, "shared": self.shared}

    def load_state_dict(self, sd):
        self.high.load_state_dict(sd["high"])
        for a, s in zip(self.low[:len(sd["low"])], sd["low"]):
            a.load_state_dict(s)


def train_hrl(cfg: dict, fleet: str, network: str, seed: int, out_dir: pathlib.Path,
              episodes: int | None = None, log_every: int = 25) -> dict:
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = episodes or cfg["training"]["episodes"]
    env = ChargingEnv(cfg, network, q_control=False)
    ctl = HRL(cfg, env.n_agg, seed)
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
        for ag in {ctl.high, *ctl.low}:
            ag.noise = noise_schedule(cfg, ep)
        spec = make_episode(cfg, fleet, "train", cur.n_ev(rng), seed=10_000 * (seed + 1) + ep)
        env.reset(spec)
        ctl.reset(env)
        hi_prev, hi_r = None, 0.0
        pending = [None] * env.n_agg
        t0 = time.time()
        i = 0
        while not env.done:
            if i % ctl.period == 0:
                s_hi = env.l1_obs()
                if hi_prev is not None:
                    ctl.high.store(hi_prev[0], hi_prev[1], hi_r, s_hi, False)
                    ctl.high.update()
                ctl.goals = ctl.high.act(s_hi, True)
                hi_prev, hi_r = (s_hi, ctl.goals.copy()), 0.0
            obs = [ctl.low_obs(env, k) for k in range(env.n_agg)]
            for k in range(env.n_agg):
                if pending[k] is not None:
                    ctl.low[k].store(*pending[k], obs[k], False)
                    ctl.low[k].update()
            env.set_corridor(*neutral_corridor(env))
            acts = [ctl.low[k].act(obs[k], True) for k in range(env.n_agg)]
            for k in range(env.n_agg):
                env.set_aggregate(k, acts[k][0], acts[k][1])
            info = env.run_interval()
            r = aggregator_rewards(cfg, env, info, mult.w)
            vp = voltage_penalty(env, info)
            hi_r += float(r.sum() + vp)
            for k in range(env.n_agg):
                u = float(info["applied_u"][k])
                rl = float(r[k] + vp - GOAL_WEIGHT * abs(u - ctl.goals[k]))
                pending[k] = (obs[k], np.array([u, acts[k][1]], np.float32), rl)
            i += 1
        for k in range(env.n_agg):
            ctl.low[k].store(*pending[k], ctl.low_obs(env, k), True)
            ctl.low[k].update()
        ctl.high.store(hi_prev[0], hi_prev[1], hi_r, env.l1_obs(), True)
        ctl.high.update()
        m = env.episode_metrics()
        cur.report(m)
        row = {"episode": ep, "stage": cur.idx, "n_ev": env.n_ev, **m, "w_retail": mult.w,
               "wall_s": round(time.time() - t0, 3)}
        mult.update(m)
        log.write(row)
        state.maybe_save(ep + 1, episodes, {"ctl": ctl, "cur": cur, "rng": rng, "mult": mult})
        if ep % log_every == 0:
            print(f"[hrl/{fleet}/{network}/s{seed}] ep{ep:4d} stage{cur.idx} cost={m['cost_eur']:.1f} "
                  f"SQ={m['service_quality']:.3f} viol={m['violation_rate_pct']:.2f}%", flush=True)
    log.close()
    torch.save({"model": ctl.state_dict(), "seed": seed, "fleet": fleet, "network": network,
                "train_wall_s": time.time() - t_start}, out_dir / "model.pt")
    state.finish()
    return {"wall_s": time.time() - t_start}
