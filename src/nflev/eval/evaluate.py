"""Frozen-policy evaluation on held-out days.

Episode j of a scenario uses seed = seed_base + j and the j-th day of a fixed
permutation of the evaluation year, so every method sees exactly the same 50
days, fleets and noise realisations (paired design).
"""
from __future__ import annotations

import csv
import pathlib
import time

from ..env.charging_env import ChargingEnv
from ..env.episode import make_episode
from .methods import build
from .runner import run_policy_episode


def evaluate(method: str, cfg: dict, scenario: str, fleet: str = "residential",
             network: str = "ieee33", split: str = "test", episodes: int | None = None,
             checkpoint: pathlib.Path | None = None, extra: dict | None = None) -> list[dict]:
    ev = cfg["evaluation"]
    episodes = episodes or ev["episodes"]
    scen = dict(ev["scenarios"][scenario])
    n_ev = scen.pop("n_ev")
    policy, q_on, env_cfg = build(method, cfg, cfg["aggregators"]["n"], checkpoint)
    env = ChargingEnv(env_cfg, network, q_control=q_on)
    mods = dict(scen)
    if getattr(policy, "ablation", None) == "no_behavior":
        mods["disable_behavior"] = True
    rows = []
    for j in range(episodes):
        seed = ev["seed_base"] + j
        spec = make_episode(env_cfg, fleet, split, n_ev, seed=seed, scenario=mods, eval_index=j)
        t0 = time.time()
        m = run_policy_episode(env, policy, spec)
        m.update(method=method, scenario=scenario, fleet=fleet, network=network, split=split,
                 episode=j, seed=seed, wall_s=round(time.time() - t0, 3), **(extra or {}))
        rows.append(m)
    return rows


def write_rows(rows: list[dict], path: pathlib.Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)
