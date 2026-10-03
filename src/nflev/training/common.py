"""Shared reward definitions and helpers for every learned method.

Aggregator reward (one per aggregator k, per 15-min interval):
    r_k = w_m (p_exec,k E_del,k - C_k) - w_u U_k - w_g G_k - w_c X_k
  E_del delivered energy, C wholesale cost of the energy drawn, U unmet energy of
  vehicles that left during the interval, G energy that can no longer be
  delivered before departure (urgency shortfall of plugged-in vehicles), X
  curtailed energy. All energies in kWh, money in EUR.
Voltage penalty (methods without Level 3 only, shared by all aggregators):
    r_V = -(10 1[V_min < 0.95] + 100 max(0, 0.95 - V_min))
Level-1 reward (hourly):
    r_1 = -w_cost C_h/100 - w_vdef (20 vdef)^2 - w_vind 1[viol] - w_unmet U_h/100 - w_curt X_h/100
"""
from __future__ import annotations

import csv
import pathlib

import numpy as np
import torch


def shortfall_by_agg(env) -> np.ndarray:
    need = env.need()
    left = np.maximum(env.dep - env.t_h, 0.0)
    gap = np.maximum(need - env.eff * env.p_max * left, 0.0)
    gap = np.where(env.connected, gap, 0.0)
    return np.bincount(env.agg, weights=gap, minlength=env.n_agg)


def aggregator_rewards(cfg: dict, env, info: dict) -> np.ndarray:
    """mode "margin":    w_m (p_exec E_del - C)   (plain retail margin)
    mode "advantage": w_m [ (p_bar - lambda_t) E_drawn + (p_exec - lambda_ref) E_del ]
    where p_bar is the mean day-ahead price over the look-ahead window. The two
    modes differ only by terms that are constant once all energy is delivered,
    but "advantage" removes the incentive to charge early that discounting
    creates under the plain margin."""
    w = cfg["level2"]["reward"]
    per = info["per"]
    if w.get("mode", "margin") == "advantage":
        p_bar = info["price_ahead_mean"] / 1000.0
        lam = info["price_now"] / 1000.0
        margin = (p_bar - lam) * per["drawn_kwh"] + (info["exec_price"] - cfg["behavior"]["lambda_ref"]) * per["delivered_kwh"]
    else:
        margin = info["exec_price"] * per["delivered_kwh"] - per["cost"]
    return (w["margin"] * margin - w["unmet"] * per["unmet_kwh"]
            - w["urgency"] * shortfall_by_agg(env) - w["curt"] * per["curtailed_kwh"])


def voltage_penalty(env, info: dict) -> float:
    vmin = info["vmin"]
    return -(10.0 * float(vmin < env.v_min - 1e-9) + 100.0 * max(0.0, env.v_min - vmin))


def voltage_cost(env, info: dict) -> float:
    vmin = info["vmin"]
    return float(vmin < env.v_min - 1e-9) + 10.0 * max(0.0, env.v_min - vmin)


class HourAccumulator:
    def __init__(self):
        self.reset()

    def reset(self):
        self.cost = self.unmet = self.curt = 0.0
        self.vmin = np.inf

    def add(self, info: dict):
        self.cost += info["cost"]
        self.unmet += float(info["per"]["unmet_kwh"].sum())
        self.curt += info["curtailed_kwh"]
        self.vmin = min(self.vmin, info["vmin"])


def level1_reward(cfg: dict, env, acc: HourAccumulator) -> float:
    w = cfg["level1"]["reward"]
    vdef = max(0.0, env.v_min - acc.vmin)
    return float(-w["cost"] * acc.cost / 100.0 - w["vdef"] * (20.0 * vdef) ** 2
                 - w["vind"] * float(acc.vmin < env.v_min - 1e-9)
                 - w["unmet"] * acc.unmet / 100.0 - w["curt"] * acc.curt / 100.0)


def corridor_from_action(env, a: np.ndarray) -> tuple[float, float]:
    span = env.price_ceil - env.price_floor - env.min_width
    p_min = env.price_floor + float(a[0]) * span
    p_max = p_min + env.min_width + float(a[1]) * (env.price_ceil - p_min - env.min_width)
    return p_min, p_max


def noise_schedule(cfg: dict, episode: int) -> float:
    p = cfg["training"]["ddpg"]
    return max(p["noise_end"], p["noise_start"] * p["noise_decay"] ** episode)


class Curriculum:
    """Advance to the next fleet size once the last `window` training episodes
    reach mean service quality >= sq_min with mean curtailment <= curt_max_kwh.
    The final stage samples the fleet size uniformly from its range."""

    def __init__(self, cfg: dict, enabled: bool = True):
        c = cfg["curriculum"]
        self.stages = c["stages"]
        self.window, self.sq_min, self.curt_max = c["window"], c["sq_min"], c["curt_max_kwh"]
        self.idx = 0 if enabled else len(self.stages) - 1
        self.hist: list[dict] = []

    def n_ev(self, rng: np.random.Generator) -> int:
        st = self.stages[self.idx]
        if "n_ev_range" in st:
            lo, hi = st["n_ev_range"]
            return int(rng.integers(lo, hi + 1))
        return int(st["n_ev"])

    def report(self, m: dict) -> bool:
        self.hist.append(m)
        if self.idx >= len(self.stages) - 1 or len(self.hist) < self.window:
            return False
        w = self.hist[-self.window:]
        if (np.mean([x["service_quality"] for x in w]) >= self.sq_min
                and np.mean([x["curtailed_kwh"] for x in w]) <= self.curt_max):
            self.idx += 1
            self.hist = []
            return True
        return False


class TrainState:
    """Crash-safe training. Every `every` episodes the complete training state
    (networks, optimizers, replay buffers, curriculum and RNG states) is pickled
    to out_dir/checkpoint.pt; a restarted run resumes from it and continues as
    if it had not been interrupted. The file is deleted when training ends."""

    def __init__(self, out_dir: pathlib.Path, every: int = 50):
        self.path = out_dir / "checkpoint.pt"
        self.every = every

    def load(self) -> dict | None:
        if not self.path.exists():
            return None
        st = torch.load(self.path, map_location="cpu", weights_only=False)
        torch.set_rng_state(st["torch_rng"])
        return st

    def maybe_save(self, ep_done: int, total: int, objs: dict) -> None:
        if ep_done % self.every or ep_done >= total:
            return
        tmp = self.path.with_suffix(".tmp")
        torch.save({"ep": ep_done, "torch_rng": torch.get_rng_state(), **objs}, tmp)
        tmp.replace(self.path)

    def finish(self) -> None:
        self.path.unlink(missing_ok=True)


class TrainLog:
    """train_log.csv writer that keeps the rows of a resumed run."""

    def __init__(self, path: pathlib.Path, resume_ep: int = 0):
        rows = []
        if resume_ep > 0 and path.exists():
            with open(path, newline="") as f:
                rows = [r for r in csv.DictReader(f) if int(r["episode"]) < resume_ep]
        self.f = open(path, "w", newline="")
        self.w = None
        for r in rows:
            self.write(r)

    def write(self, row: dict) -> None:
        if self.w is None:
            self.w = csv.DictWriter(self.f, fieldnames=list(row))
            self.w.writeheader()
        self.w.writerow(row)
        self.f.flush()

    def close(self) -> None:
        self.f.close()
