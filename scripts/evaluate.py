#!/usr/bin/env python3
"""Evaluate trained checkpoints and baselines across scenarios S1-S7.

Usage:
  python scripts/evaluate.py --methods uncoordinated tou milp nested \
      --scenarios S1 S2 S3 S4 S5 S6_deadlock S7_q_exhaustion \
      --checkpoint results/nested_ieee33_nrel_s0 --episodes 50 \
      --network ieee33 --dataset nrel --out results/eval_ieee33_nrel.csv
"""
import argparse, csv, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import yaml

from nflev.env.charging_env import ChargingEnv
from nflev.baselines.simple import UncoordinatedPolicy, TOUPolicy, run_baseline_episode
from nflev.baselines.milp import run_milp_episode
from nflev.baselines.flat_ddpg import FlatDDPG
from nflev.baselines.hrl import HRLBaseline
from nflev.agents.ppo_lagrangian import PPOLagrangian
from nflev.agents.cpo import CPO
from nflev.training.trainer import NestedTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOAD_SHAPE = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])


def price_profile(source: str) -> np.ndarray:
    f = ROOT / f"configs/prices_{source}.yaml"
    if f.exists():
        return np.array(yaml.safe_load(f.read_text())["hourly_lmp_usd_mwh"], float)
    return np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float)


def make_env(cfg, network, ds, scen: dict, seed: int, disable_q=False):
    mods = {k: v for k, v in scen.items() if k not in ("penetration", "n_ev")}
    if disable_q:
        mods["disable_q_control"] = True
    return ChargingEnv(cfg, network, ds, n_ev=scen["n_ev"], seed=seed,
                       scenario_mods=mods)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+",
                    default=["uncoordinated", "tou", "milp", "nested"],
                    help="uncoordinated tou milp nested flat_ddpg ppo_lag cpo hrl")
    ap.add_argument("--baseline_ckpts", default="results",
                    help="dir containing {method}_{network}_{dataset}_s0/final[.pt]")
    ap.add_argument("--scenarios", nargs="+", default=["S3"])
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--dataset", default="nrel")
    ap.add_argument("--out", default="results/eval.csv")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ds = yaml.safe_load(open(ROOT / f"configs/dataset/{args.dataset}.yaml"))
    price = price_profile(ds.get("price_source", "pjm"))
    scen_defs = cfg["evaluation"]["scenarios"]

    learners = {}
    for meth, cls in [("flat_ddpg", FlatDDPG), ("ppo_lag", PPOLagrangian),
                      ("cpo", CPO), ("hrl", HRLBaseline)]:
        if meth in args.methods:
            ag = cls(cfg)
            if meth in ("ppo_lag", "cpo"):
                ag._cfg = cfg
            ck = pathlib.Path(args.baseline_ckpts) / f"{meth}_{args.network}_{args.dataset}_s0"
            ag.load(str(ck / "final") if meth == "hrl" else str(ck / "final.pt"))
            learners[meth] = ag

    trainer = None
    if "nested" in args.methods:
        assert args.checkpoint, "--checkpoint required for nested"
        ck = pathlib.Path(args.checkpoint)
        trainer = NestedTrainer(cfg, lambda st, sd: None)
        trainer.l1.load(ck / "l1_final.pt")
        for k, ag in enumerate(trainer.l2):
            ag.load(ck / f"l2_{k}_final.pt")

    out = pathlib.Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "scenario", "episode", "daily_cost_usd",
              "min_voltage_pu", "violation_rate_pct", "service_quality",
              "q_activation_freq", "curtailed_kwh", "q_exhausted_any",
              "peak_mw", "infeasible", "wall_s"]
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for scen_name in args.scenarios:
            scen = scen_defs[scen_name]
            for method in args.methods:
                for ep in range(args.episodes):
                    seed = 7000 + ep
                    t0 = time.time()
                    if method == "uncoordinated":
                        env = make_env(cfg, args.network, ds, scen, seed, disable_q=True)
                        m = run_baseline_episode(env, UncoordinatedPolicy(),
                                                 price, LOAD_SHAPE)
                    elif method == "tou":
                        env = make_env(cfg, args.network, ds, scen, seed, disable_q=True)
                        m = run_baseline_episode(env, TOUPolicy(), price, LOAD_SHAPE)
                    elif method == "milp":
                        env = make_env(cfg, args.network, ds, scen, seed, disable_q=True)
                        m = run_milp_episode(env, price, LOAD_SHAPE)
                    elif method == "nested":
                        env = make_env(cfg, args.network, ds, scen, seed)
                        trainer.env_factory = lambda st, sd, e=env: e
                        m = trainer.run_episode(seed=seed, train=False,
                                                scenario=scen,
                                                price_profile=price,
                                                load_profile=LOAD_SHAPE)
                    elif method in ("flat_ddpg", "hrl"):
                        env = make_env(cfg, args.network, ds, scen, seed, disable_q=True)
                        m = learners[method].run_episode(env, price, LOAD_SHAPE, train=False)
                    elif method in ("ppo_lag", "cpo"):
                        from train_baseline import run_onpolicy_episode
                        env = make_env(cfg, args.network, ds, scen, seed, disable_q=True)
                        m = run_onpolicy_episode(learners[method], env, price,
                                                 LOAD_SHAPE, train=False)
                    else:
                        raise ValueError(method)
                    m.update(method=method, scenario=scen_name, episode=ep,
                             wall_s=round(time.time() - t0, 1))
                    w.writerow(m); f.flush()
                print(f"[{scen_name}/{method}] done {args.episodes} episodes", flush=True)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()