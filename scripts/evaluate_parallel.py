#!/usr/bin/env python3
"""Parallel evaluation across CPU cores. Same outputs as evaluate.py but fans
(method, scenario, episode) jobs across a process pool. On the 8-core node use
--workers 7 (leave one core for the OS); ~7x throughput on eval sweeps.

Usage:
  python scripts/evaluate_parallel.py --methods uncoordinated tou milp nested \
      flat_ddpg ppo_lag cpo hrl --scenarios S1 S2 S3 S4 S5 --episodes 50 \
      --checkpoint results/nested_ieee33_nrel_s0 --workers 7 \
      --out results/eval_ieee33_nrel.csv

Learning-baseline checkpoints are loaded once per worker (lazy, cached) rather
than per episode. Nested/MILP/simple methods need no warm state.
"""
import argparse, csv, os, pathlib, sys, time
from concurrent.futures import ProcessPoolExecutor, as_completed
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "scripts"))
import numpy as np
import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOAD_SHAPE = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])

_CACHE = {}  # per-worker: (method, ck) -> agent/trainer


def _price(source):
    f = ROOT / f"configs/prices_{source}.yaml"
    if f.exists():
        return np.array(yaml.safe_load(f.read_text())["hourly_lmp_usd_mwh"], float)
    return np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float)


def _run_one(job):
    method, scen_name, ep, network, dataset, checkpoint, bl_dir = job
    import yaml as _yaml
    from nflev.env.charging_env import ChargingEnv
    cfg = _yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ds = _yaml.safe_load(open(ROOT / f"configs/dataset/{dataset}.yaml"))
    price = _price(ds.get("price_source", "pjm"))
    scen = cfg["evaluation"]["scenarios"][scen_name]
    seed = 7000 + ep
    mods = {k: v for k, v in scen.items() if k not in ("penetration", "n_ev")}
    disable_q = method != "nested"
    if disable_q:
        mods["disable_q_control"] = True
    env = ChargingEnv(cfg, network, ds, n_ev=scen["n_ev"], seed=seed,
                      scenario_mods=mods)
    t0 = time.time()

    if method == "uncoordinated":
        from nflev.baselines.simple import UncoordinatedPolicy, run_baseline_episode
        m = run_baseline_episode(env, UncoordinatedPolicy(), price, LOAD_SHAPE)
    elif method == "tou":
        from nflev.baselines.simple import TOUPolicy, run_baseline_episode
        m = run_baseline_episode(env, TOUPolicy(), price, LOAD_SHAPE)
    elif method == "milp":
        from nflev.baselines.milp import run_milp_episode
        m = run_milp_episode(env, price, LOAD_SHAPE)
    else:
        key = (method, checkpoint or bl_dir)
        if key not in _CACHE:
            _CACHE[key] = _load_agent(method, cfg, network, dataset,
                                      checkpoint, bl_dir)
        agent = _CACHE[key]
        if method == "nested":
            agent.env_factory = lambda st, sd, e=env: e
            m = agent.run_episode(seed=seed, train=False, scenario=scen,
                                  price_profile=price, load_profile=LOAD_SHAPE)
        elif method in ("flat_ddpg", "hrl"):
            m = agent.run_episode(env, price, LOAD_SHAPE, train=False)
        else:  # ppo_lag, cpo
            from train_baseline import run_onpolicy_episode
            m = run_onpolicy_episode(agent, env, price, LOAD_SHAPE, train=False)
    m.update(method=method, scenario=scen_name, episode=ep,
             wall_s=round(time.time() - t0, 1))
    return m


def _load_agent(method, cfg, network, dataset, checkpoint, bl_dir):
    if method == "nested":
        from nflev.training.trainer import NestedTrainer
        t = NestedTrainer(cfg, lambda st, sd: None)
        ck = pathlib.Path(checkpoint)
        t.l1.load(ck / "l1_final.pt")
        for k, ag in enumerate(t.l2):
            ag.load(ck / f"l2_{k}_final.pt")
        return t
    from nflev.baselines.flat_ddpg import FlatDDPG
    from nflev.baselines.hrl import HRLBaseline
    from nflev.agents.ppo_lagrangian import PPOLagrangian
    from nflev.agents.cpo import CPO
    cls = {"flat_ddpg": FlatDDPG, "hrl": HRLBaseline,
           "ppo_lag": PPOLagrangian, "cpo": CPO}[method]
    ag = cls(cfg)
    if method in ("ppo_lag", "cpo"):
        ag._cfg = cfg
    ck = pathlib.Path(bl_dir) / f"{method}_{network}_{dataset}_s0"
    ag.load(str(ck / "final") if method == "hrl" else str(ck / "final.pt"))
    return ag


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--methods", nargs="+", required=True)
    ap.add_argument("--scenarios", nargs="+", required=True)
    ap.add_argument("--episodes", type=int, default=50)
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--dataset", default="nrel")
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--baseline_ckpts", default="results")
    ap.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 1))
    ap.add_argument("--out", default="results/eval_parallel.csv")
    args = ap.parse_args()

    jobs = [(m, s, ep, args.network, args.dataset, args.checkpoint,
             args.baseline_ckpts)
            for s in args.scenarios for m in args.methods
            for ep in range(args.episodes)]
    print(f"{len(jobs)} jobs across {args.workers} workers")

    out = pathlib.Path(args.out); out.parent.mkdir(parents=True, exist_ok=True)
    fields = ["method", "scenario", "episode", "daily_cost_usd",
              "min_voltage_pu", "violation_rate_pct", "service_quality",
              "q_activation_freq", "curtailed_kwh", "q_exhausted_any",
              "peak_mw", "infeasible", "wall_s"]
    done = 0
    t0 = time.time()
    with open(out, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        with ProcessPoolExecutor(max_workers=args.workers) as ex:
            futs = [ex.submit(_run_one, j) for j in jobs]
            for fut in as_completed(futs):
                w.writerow(fut.result()); f.flush()
                done += 1
                if done % 25 == 0:
                    rate = done / (time.time() - t0)
                    print(f"  {done}/{len(jobs)} ({rate:.1f} ep/s)", flush=True)
    print(f"wrote {out} in {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()