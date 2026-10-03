"""Nested framework: training loop and deterministic controller.

Level 1 (PPO, hourly) sets the retail corridor from an 11-d system state.
Level 2 (one DDPG agent per aggregator, every 15 min) outputs an aggregate
power fraction u_k and an execution-price fraction phi_k from a 27-d local
state (voltage, time, current and 12 h of day-ahead prices, corridor, energy
needs binned by laxity, ...). The environment turns u_k into per-vehicle rates
by least-laxity-first allocation with a deadline guard (feasible by
construction). Level 3 (non-parametric) is part of the environment.

Ablations: none | no_l1 | flat_timescale | no_behavior | no_l3 | no_curriculum |
           no_guard | proportional (proportional instead of LLF disaggregation)
"""
from __future__ import annotations

import copy
import csv
import pathlib
import time

import numpy as np
import torch

from ..agents.ddpg import DDPGAgent
from ..agents.ppo import PPOAgent
from ..env.charging_env import L1_OBS_DIM, L2_OBS_DIM, ChargingEnv
from ..env.episode import make_episode
from .common import (Curriculum, HourAccumulator, aggregator_rewards, corridor_from_action,
                     level1_reward, noise_schedule, voltage_penalty)

ABLATIONS = ("none", "no_l1", "flat_timescale", "no_behavior", "no_l3", "no_curriculum",
             "no_guard", "proportional")


def ablated_cfg(cfg: dict, ablation: str) -> dict:
    c = copy.deepcopy(cfg)
    if ablation == "no_guard":
        c["level2"]["deadline_guard"] = False
    if ablation == "proportional":
        c["level2"]["disaggregation"] = "proportional"
    return c


class NestedController:
    """Deterministic nested policy (evaluation); also used during training."""
    name = "nested"

    def __init__(self, cfg: dict, n_agg: int, seed: int, ablation: str = "none"):
        if ablation not in ABLATIONS:
            raise ValueError(ablation)
        self.cfg, self.ablation = cfg, ablation
        self.n_agg = n_agg
        self.shared = bool(cfg["level2"].get("shared", False))
        self.l1 = PPOAgent(cfg, L1_OBS_DIM, 2)
        if self.shared:      # one actor-critic for all aggregators, aggregator id one-hot in the state
            agent = DDPGAgent(cfg, L2_OBS_DIM + n_agg, 2, seed * 100)
            self.l2 = [agent] * n_agg
        else:
            self.l2 = [DDPGAgent(cfg, L2_OBS_DIM, 2, seed * 100 + k) for k in range(n_agg)]
        self.explore = False

    def l2_input(self, env, k: int) -> np.ndarray:
        o = env.l2_obs(k)
        if self.shared:
            o = np.concatenate([o, np.eye(self.n_agg, dtype=np.float32)[k]])
        return o

    def l1_period(self, env) -> int:
        return 1 if self.ablation == "flat_timescale" else \
            int(env.cfg["simulation"]["pricing_interval_s"] // env.cfg["simulation"]["dispatch_interval_s"])

    def reset(self, env) -> None:
        self.interval = 0

    def l1_act(self, env):
        if self.ablation == "no_l1":
            env.set_corridor(env.price_floor, env.price_ceil)
            return None
        s = env.l1_obs()
        a, logp = self.l1.act(s, self.explore)
        env.set_corridor(*corridor_from_action(env, a))
        return s, a, logp

    def act(self, env) -> None:
        if self.interval % self.l1_period(env) == 0:
            self.l1_act(env)
        for k in range(env.n_agg):
            a = self.l2[k].act(self.l2_input(env, k), self.explore)
            env.set_aggregate(k, a[0], a[1])
        self.interval += 1

    def state_dict(self) -> dict:
        l2 = [self.l2[0].state_dict()] if self.shared else [ag.state_dict() for ag in self.l2]
        return {"l1": self.l1.state_dict(), "l2": l2, "ablation": self.ablation, "shared": self.shared}

    def load_state_dict(self, sd: dict) -> None:
        self.l1.load_state_dict(sd["l1"])
        for ag, s in zip(self.l2[:len(sd["l2"])], sd["l2"]):
            ag.load_state_dict(s)


def episode_mods(ablation: str) -> dict:
    return {"disable_behavior": True} if ablation == "no_behavior" else {}


def train_nested(cfg: dict, fleet: str, network: str, seed: int, ablation: str,
                 out_dir: pathlib.Path, episodes: int | None = None, log_every: int = 25) -> dict:
    cfg = ablated_cfg(cfg, ablation)
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)
    episodes = episodes or cfg["training"]["episodes"]
    env = ChargingEnv(cfg, network, q_control=(ablation != "no_l3"))
    ctl = NestedController(cfg, env.n_agg, seed, ablation)
    ctl.explore = True
    cur = Curriculum(cfg, enabled=(ablation != "no_curriculum"))
    out_dir.mkdir(parents=True, exist_ok=True)
    log_f = open(out_dir / "train_log.csv", "w", newline="")
    log = None
    t_start = time.time()
    for ep in range(episodes):
        for ag in set(ctl.l2):
            ag.noise = noise_schedule(cfg, ep)
        spec = make_episode(cfg, fleet, "train", cur.n_ev(rng), seed=10_000 * (seed + 1) + ep,
                            scenario=episode_mods(ablation))
        env.reset(spec)
        ctl.reset(env)
        period = ctl.l1_period(env)
        n_int = int(round(spec.horizon_h / env.interval_h))
        pending = [None] * env.n_agg                 # (s, a_applied, r) waiting for s'
        l1_prev, acc = None, HourAccumulator()
        t0 = time.time()
        for i in range(n_int):
            if i % period == 0:
                if l1_prev is not None:
                    ctl.l1.store(*l1_prev, level1_reward(cfg, env, acc), False)
                l1_prev = ctl.l1_act(env)
                acc.reset()
            obs = [ctl.l2_input(env, k) for k in range(env.n_agg)]
            for k in range(env.n_agg):
                if pending[k] is not None:
                    s, a, r = pending[k]
                    ctl.l2[k].store(s, a, r, obs[k], False)
                    ctl.l2[k].update()
            acts = [ctl.l2[k].act(obs[k], True) for k in range(env.n_agg)]
            for k in range(env.n_agg):
                env.set_aggregate(k, acts[k][0], acts[k][1])
            info = env.run_interval()
            acc.add(info)
            r = aggregator_rewards(cfg, env, info)
            if ablation == "no_l3":
                r = r + voltage_penalty(env, info)
            for k in range(env.n_agg):
                applied = np.array([info["applied_u"][k], acts[k][1]], np.float32)
                pending[k] = (obs[k], applied, float(r[k]))
        for k in range(env.n_agg):
            s, a, rr = pending[k]
            ctl.l2[k].store(s, a, rr, ctl.l2_input(env, k), True)
            ctl.l2[k].update()
        if l1_prev is not None:
            ctl.l1.store(*l1_prev, level1_reward(cfg, env, acc), True)
            ctl.l1.update()
        m = env.episode_metrics()
        advanced = cur.report(m)
        row = {"episode": ep, "stage": cur.idx, "n_ev": env.n_ev, **m,
               "noise": ctl.l2[0].noise, "wall_s": round(time.time() - t0, 3)}
        if log is None:
            log = csv.DictWriter(log_f, fieldnames=list(row))
            log.writeheader()
        log.writerow(row)
        log_f.flush()
        if ep % log_every == 0 or advanced:
            print(f"[{ablation}/{fleet}/{network}/s{seed}] ep{ep:4d} stage{cur.idx} n_ev={env.n_ev} "
                  f"cost={m['cost_eur']:.1f} SQ={m['service_quality']:.3f} viol={m['violation_rate_pct']:.2f}% "
                  f"curt={m['curtailed_kwh']:.1f} {'ADVANCED' if advanced else ''}", flush=True)
    log_f.close()
    ctl.explore = False
    torch.save({**ctl.state_dict(), "fleet": fleet, "network": network, "seed": seed,
                "episodes": episodes, "final_stage": cur.idx,
                "train_wall_s": time.time() - t_start}, out_dir / "model.pt")
    return {"final_stage": cur.idx, "wall_s": time.time() - t_start}


def load_nested(path: pathlib.Path, cfg: dict) -> NestedController:
    sd = torch.load(path, map_location="cpu", weights_only=False)
    c = ablated_cfg(cfg, sd["ablation"])
    c["level2"]["shared"] = sd.get("shared", False)
    ctl = NestedController(c, cfg["aggregators"]["n"], 0, sd["ablation"])
    ctl.load_state_dict(sd)
    ctl.explore = False
    return ctl
