"""Shared reward definitions and helpers for every learned method.

Aggregator reward (one per aggregator k, per 15-min interval), mode "cost":
    r_k = -C_k + w (p_exec,k - lambda_ref) E_del,k - w_u U_k - w_g G_k - w_c X_k
  C wholesale cost of the energy drawn, E_del delivered energy, U unmet energy of
  vehicles that left during the interval, G energy that can no longer be
  delivered before departure (urgency shortfall of plugged-in vehicles), X
  curtailed energy. All energies in kWh, money in EUR. Summed over an episode
  the first term is exactly minus the energy cost. w is the retail weight; with
  level2.revenue_neutral it is a Lagrange multiplier that keeps the mean retail
  price paid at lambda_ref (see RetailMultiplier).
Voltage penalty (methods without Level 3 only, shared by all aggregators):
    r_V = -(10 1[V_min < 0.95] + 100 max(0, 0.95 - V_min))
Level-1 reward (hourly):
    r_1 = -w_cost C_h/100 - w_vdef (20 vdef)^2 - w_vind 1[viol] - w_unmet U_h/100 - w_curt X_h/100
          + w (R_h - lambda_ref E_del,h)/100       (R_h retail revenue of the hour)
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


def aggregator_rewards(cfg: dict, env, info: dict, w_retail: float | None = None) -> np.ndarray:
    """mode "cost":      -C + w (p_exec - lambda_ref) E_del          (default)
    mode "margin":    w (p_exec E_del - C)                         (plain retail margin)
    mode "advantage": w [(p_bar - lambda_t) E_drawn + (p_exec - lambda_ref) E_del]
    (p_bar: mean day-ahead price over the look-ahead window). "advantage" was used
    in earlier versions; it credits any energy drawn below the look-ahead mean
    and therefore favours afternoon charging when the cheapest hours lie beyond
    the window."""
    w = cfg["level2"]["reward"]
    wr = w["margin"] if w_retail is None else w_retail
    per = info["per"]
    lam_ref = cfg["behavior"]["lambda_ref"]
    mode = w.get("mode", "margin")
    if mode == "cost":
        margin = -per["cost"] + wr * (info["exec_price"] - lam_ref) * per["delivered_kwh"]
    elif mode == "advantage":
        p_bar = info["price_ahead_mean"] / 1000.0
        lam = info["price_now"] / 1000.0
        margin = wr * ((p_bar - lam) * per["drawn_kwh"] + (info["exec_price"] - lam_ref) * per["delivered_kwh"])
    else:
        margin = wr * (info["exec_price"] * per["delivered_kwh"] - per["cost"])
    return (margin - w["unmet"] * per["unmet_kwh"]
            - w["urgency"] * shortfall_by_agg(env) - w["curt"] * per["curtailed_kwh"])


class RetailMultiplier:
    """Revenue-neutral dynamic tariff. The weight w of the retail term is the
    Lagrange multiplier of the constraint  mean retail price paid = lambda_ref,
    set once per training episode by a PI Lagrangian update (Stooke et al., 2020),
    which damps the oscillations of plain dual ascent:
        e     = (lambda_ref - p_paid) / lambda_ref        (> 0: prices too low)
        e_bar <- (1 - a) e_bar + a e                       (smoothed error)
        I     <- clip(I + e, -I_max, I_max)                (anti-windup)
        w     =  clip(K_p e_bar + K_i I, w_min, w_max)."""

    def __init__(self, cfg: dict):
        rn = cfg["level2"].get("revenue_neutral", {}) or {}
        self.enabled = bool(rn.get("enabled", False))
        self.kp, self.ki = float(rn.get("kp", 0.0)), float(rn.get("ki", 0.1))
        self.a = float(rn.get("ema", 1.0))
        self.w_min, self.w_max = float(rn.get("w_min", 0.0)), float(rn.get("w_max", 5.0))
        self.ref = float(cfg["behavior"]["lambda_ref"])
        self.w = float(rn.get("w_init", 0.0)) if self.enabled else float(cfg["level2"]["reward"]["margin"])
        self.e_bar = 0.0
        self.integral = self.w / self.ki if self.ki > 0 else 0.0
        self.i_max = max(abs(self.w_min), abs(self.w_max)) / self.ki if self.ki > 0 else 0.0

    def update(self, m: dict) -> None:
        p = m.get("retail_price_paid", float("nan"))
        if not (self.enabled and np.isfinite(p)):
            return
        e = (self.ref - p) / self.ref
        self.e_bar = (1.0 - self.a) * self.e_bar + self.a * e
        self.integral = float(np.clip(self.integral + e, -self.i_max, self.i_max))
        self.w = float(np.clip(self.kp * self.e_bar + self.ki * self.integral, self.w_min, self.w_max))


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
        self.cost = self.unmet = self.curt = self.revenue = self.delivered = 0.0
        self.vmin = np.inf

    def add(self, info: dict):
        self.cost += info["cost"]
        self.revenue += float((info["exec_price"] * info["per"]["delivered_kwh"]).sum())
        self.delivered += float(info["per"]["delivered_kwh"].sum())
        self.unmet += float(info["per"]["unmet_kwh"].sum())
        self.curt += info["curtailed_kwh"]
        self.vmin = min(self.vmin, info["vmin"])


def level1_reward(cfg: dict, env, acc: HourAccumulator, w_retail: float = 0.0) -> float:
    w = cfg["level1"]["reward"]
    vdef = max(0.0, env.v_min - acc.vmin)
    retail = w_retail * (acc.revenue - cfg["behavior"]["lambda_ref"] * acc.delivered) / 100.0
    return float(-w["cost"] * acc.cost / 100.0 - w["vdef"] * (20.0 * vdef) ** 2
                 - w["vind"] * float(acc.vmin < env.v_min - 1e-9)
                 - w["unmet"] * acc.unmet / 100.0 - w["curt"] * acc.curt / 100.0 + retail)


def corridor_from_action(env, a: np.ndarray) -> tuple[float, float]:
    """Level-1 action in [0, 1]^2 -> retail corridor [p_min, p_max].
    a_1 sets the centre: piecewise linear with a_1 = 0.5 at the reference rate
    lambda_ref, 0 at the price floor and 1 at the ceiling. a_2 sets the width
    between min_corridor_width and max_corridor_width. A neutral action
    (0.5, 0.5) therefore gives a corridor centred on the flat reference tariff,
    so an untrained policy charges what the non-pricing baselines charge."""
    ref, lo, hi = env.ref_price, env.price_floor, env.price_ceil
    a0, a1 = float(a[0]), float(a[1])
    c = ref + (2 * a0 - 1) * ((ref - lo) if a0 < 0.5 else (hi - ref))
    half = 0.5 * (env.min_width + a1 * (env.max_width - env.min_width))
    p_min = float(np.clip(c - half, lo, hi - env.min_width))
    p_max = float(np.clip(c + half, p_min + env.min_width, hi))
    return p_min, p_max


def neutral_corridor(env) -> tuple[float, float]:
    """Corridor of the neutral Level-1 action; used whenever no learned Level 1
    sets the corridor (no-Level-1 ablation, learned baselines)."""
    return corridor_from_action(env, np.array([0.5, 0.5]))


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
