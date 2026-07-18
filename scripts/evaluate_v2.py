#!/usr/bin/env python3
"""Frozen-policy evaluation across scenarios S1-S7 (campaign-grade).

Strict superset of ``scripts/evaluate.py``. Additions required to defend the
manuscript's tables:

  --ablation      Reconstruct the NestedTrainer with the same ablation flag the
                  checkpoint was trained with (no_l1 / flat_timescale), so
                  architecture and checkpoint always match.
  --env_mods      Repeatable ``key=value`` scenario-mod injections for the
                  nested method (e.g. ``disable_q_control=true`` for the
                  "No Level 3" ablation row, ``disable_behavior=true`` for the
                  "No user model" row).
  --cfg_override  Repeatable dot-path config overrides applied before the env
                  is built (e.g. ``voltage.correction_margin=0.0005``,
                  ``reactive_power.s_rated_kva=10.0``) - used by the
                  hyperparameter-sensitivity table.
  --seed_base     Base evaluation seed (default 7000 -> seeds 7000..7049 for
                  50 episodes, matching the protocol stated in the paper).

Every row written to the output CSV is one full frozen-policy episode; nothing
is read from training logs.

Usage (single cell of the main table):
  python scripts/evaluate_v2.py --methods nested --scenarios S3 \
      --checkpoint results/nested_ieee33_nrel --episodes 50 \
      --network ieee33 --dataset nrel --out results/eval/nested_s3.csv
"""
from __future__ import annotations

import argparse
import csv
import json
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))

import os
for _v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS",
           "OPENBLAS_NUM_THREADS", "NUMEXPR_NUM_THREADS"):
    os.environ.setdefault(_v, "1")

import numpy as np
import yaml

from nflev.env.charging_env import ChargingEnv
from nflev.baselines.simple import (UncoordinatedPolicy, TOUPolicy,
                                    run_baseline_episode)
from nflev.baselines.milp import run_milp_episode
from nflev.baselines.flat_ddpg import FlatDDPG
from nflev.baselines.hrl import HRLBaseline
from nflev.agents.ppo_lagrangian import PPOLagrangian
from nflev.agents.cpo import CPO
from nflev.training.trainer import NestedTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOAD_SHAPE = np.array([.55, .5, .48, .47, .48, .55, .65, .75, .72, .70, .68,
                       .67, .66, .68, .72, .80, .92, 1.0, 1.0, .97, .88, .78,
                       .68, .60])

FIELDS = ["method", "scenario", "episode", "seed", "daily_cost_usd",
          "min_voltage_pu", "violation_rate_pct", "service_quality",
          "q_activation_freq", "curtailed_kwh", "q_exhausted_any",
          "peak_mw", "infeasible", "wall_s"]


# --------------------------------------------------------------------------- #
# helpers
# --------------------------------------------------------------------------- #
def parse_kv(pairs: list[str]) -> dict:
    """Parse ``key=value`` pairs with YAML-typed values (true/1.5/text)."""
    out = {}
    for p in pairs or []:
        if "=" not in p:
            raise SystemExit(f"--env_mods/--cfg_override expects key=value, got {p!r}")
        k, v = p.split("=", 1)
        out[k.strip()] = yaml.safe_load(v)
    return out


def apply_cfg_overrides(cfg: dict, overrides: dict) -> dict:
    """Apply dot-path overrides in place; fail loudly on unknown keys."""
    for path, val in overrides.items():
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            if k not in node:
                raise SystemExit(f"cfg override path not found: {path!r}")
            node = node[k]
        if keys[-1] not in node:
            raise SystemExit(f"cfg override leaf not found: {path!r}")
        node[keys[-1]] = val
    return cfg


def price_profile(source: str) -> np.ndarray:
    f = ROOT / f"configs/prices_{source}.yaml"
    if f.exists():
        return np.array(yaml.safe_load(f.read_text())["hourly_lmp_usd_mwh"],
                        float)
    return np.array([28, 26, 25, 25, 26, 30, 45, 60, 55, 50, 48, 47, 46, 48,
                     52, 60, 80, 110, 120, 105, 85, 60, 45, 35], float)


def make_env(cfg, network, ds, scen: dict, seed: int,
             disable_q: bool = False, extra_mods: dict | None = None):
    mods = {k: v for k, v in scen.items() if k not in ("penetration", "n_ev")}
    if disable_q:
        mods["disable_q_control"] = True
    if extra_mods:
        mods.update(extra_mods)
    return ChargingEnv(cfg, network, ds, n_ev=scen["n_ev"], seed=seed,
                       scenario_mods=mods)


# --------------------------------------------------------------------------- #
# main
# --------------------------------------------------------------------------- #
def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--methods", nargs="+",
                    default=["uncoordinated", "tou", "milp", "nested"],
                    help="uncoordinated tou milp nested flat_ddpg ppo_lag cpo hrl")
    ap.add_argument("--baseline_ckpts", default="results",
                    help="dir containing {method}_{network}_{dataset}_s0/final[.pt]")
    ap.add_argument("--scenarios", nargs="+", default=["S3"])
    ap.add_argument("--checkpoint", default=None,
                    help="nested checkpoint dir (l1_final.pt, l2_k_final.pt)")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_l1", "flat_timescale"],
                    help="ablation flag the nested checkpoint was trained with")
    ap.add_argument("--env_mods", nargs="*", default=[], metavar="K=V",
                    help="scenario-mod injections for the nested env")
    ap.add_argument("--cfg_override", nargs="*", default=[], metavar="PATH=V",
                    help="dot-path config overrides (sensitivity sweeps)")
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--seed_base", type=int, default=7000)
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--dataset", default="nrel")
    ap.add_argument("--out", default="results/eval.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    cfg = apply_cfg_overrides(cfg, parse_kv(args.cfg_override))
    ds = yaml.safe_load(open(ROOT / f"configs/dataset/{args.dataset}.yaml"))
    price = price_profile(ds.get("price_source", "pjm"))
    scen_defs = cfg["evaluation"]["scenarios"]
    extra_mods = parse_kv(args.env_mods)

    # -- learned baselines ---------------------------------------------------
    learners = {}
    for meth, cls in [("flat_ddpg", FlatDDPG), ("ppo_lag", PPOLagrangian),
                      ("cpo", CPO), ("hrl", HRLBaseline)]:
        if meth in args.methods:
            ag = cls(cfg)
            if meth in ("ppo_lag", "cpo"):
                ag._cfg = cfg
            ck = (pathlib.Path(args.baseline_ckpts)
                  / f"{meth}_{args.network}_{args.dataset}_s0")
            ag.load(str(ck / "final") if meth == "hrl" else str(ck / "final.pt"))
            learners[meth] = ag

    # -- nested (architecture must match the checkpoint's ablation flag) -----
    trainer = None
    if "nested" in args.methods:
        assert args.checkpoint, "--checkpoint required for nested"
        ck = pathlib.Path(args.checkpoint)
        trainer = NestedTrainer(cfg, lambda st, sd: None, ablation=args.ablation)
        trainer.l1.load(ck / "l1_final.pt")
        for k, ag in enumerate(trainer.l2):
            ag.load(ck / f"l2_{k}_final.pt")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta = {"argv": sys.argv[1:], "cfg_override": parse_kv(args.cfg_override),
            "env_mods": extra_mods, "seed_base": args.seed_base}
    out.with_suffix(".meta.json").write_text(json.dumps(meta, indent=2))

    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS, extrasaction="ignore")
        w.writeheader()
        for scen_name in args.scenarios:
            scen = scen_defs[scen_name]
            for method in args.methods:
                for ep in range(args.episodes):
                    seed = args.seed_base + ep
                    t0 = time.time()
                    if method == "uncoordinated":
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       disable_q=True)
                        m = run_baseline_episode(env, UncoordinatedPolicy(),
                                                 price, LOAD_SHAPE)
                    elif method == "tou":
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       disable_q=True)
                        m = run_baseline_episode(env, TOUPolicy(), price,
                                                 LOAD_SHAPE)
                    elif method == "milp":
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       disable_q=True)
                        m = run_milp_episode(env, price, LOAD_SHAPE)
                    elif method == "nested":
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       extra_mods=extra_mods)
                        trainer.env_factory = lambda st, sd, e=env: e
                        m = trainer.run_episode(seed=seed, train=False,
                                                scenario=scen,
                                                price_profile=price,
                                                load_profile=LOAD_SHAPE)
                    elif method in ("flat_ddpg", "hrl"):
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       disable_q=True)
                        m = learners[method].run_episode(env, price,
                                                         LOAD_SHAPE,
                                                         train=False)
                    elif method in ("ppo_lag", "cpo"):
                        from train_baseline import run_onpolicy_episode
                        env = make_env(cfg, args.network, ds, scen, seed,
                                       disable_q=True)
                        m = run_onpolicy_episode(learners[method], env, price,
                                                 LOAD_SHAPE, train=False)
                    else:
                        raise ValueError(method)
                    m.update(method=method, scenario=scen_name, episode=ep,
                             seed=seed, wall_s=round(time.time() - t0, 1))
                    w.writerow(m)
                    f.flush()
                print(f"[{scen_name}/{method}] done {args.episodes} episodes",
                      flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()