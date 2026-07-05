#!/usr/bin/env python3
"""Train a learning baseline (flat DDPG, PPO-Lagrangian, CPO, or HRL).

Usage:
  python scripts/train_baseline.py --method ppo_lag --episodes 600 \
      --network ieee33 --dataset nrel --seed 0 --out results/ppolag_s0

All learning baselines run WITHOUT Level 3 Q-control (voltage safety must be
learned), matching the comparison protocol in the paper.
"""
import argparse, csv, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import numpy as np
import yaml

from nflev.env.charging_env import ChargingEnv
from nflev.baselines.flat_ddpg import FlatDDPG, MAX_EVS_FLAT
from nflev.baselines.hrl import HRLBaseline
from nflev.agents.ppo_lagrangian import PPOLagrangian
from nflev.agents.cpo import CPO

ROOT = pathlib.Path(__file__).resolve().parents[1]
LOAD_SHAPE = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])


def price_profile(source):
    f = ROOT / f"configs/prices_{source}.yaml"
    if f.exists():
        return np.array(yaml.safe_load(f.read_text())["hourly_lmp_usd_mwh"], float)
    return np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float)


def cost_signal(interval) -> float:
    """Constraint cost for safe-RL methods: violation indicator + graded deficit."""
    deficit = max(0.0, 0.95 - interval["v_min"])
    return float(interval["v_min"] < 0.95 - 1e-9) + 10.0 * deficit


def run_onpolicy_episode(agent, env, price, load, train=True):
    """Shared driver for PPO-Lagrangian and CPO (flat on-policy agents)."""
    env.reset(price, load)
    n_intervals = int(env.episode_h * 3600 // env.dispatch_s)
    c_max = agent_cfg_cmax
    helper = FlatDDPG(agent._cfg, n_agg=env.n_agg)  # reuse state/action mapping
    dep_seen = set()
    for i in range(n_intervals):
        s = helper.flat_state(env)
        a, raw, logp = agent.act(s, deterministic=not train)
        helper.apply_action(env, a)
        interval = env.run_dispatch_interval()
        unmet = 0.0
        for ev in env.evs:
            if ev.departed and ev.idx not in dep_seen:
                unmet += max(0.0, ev.soc_target - ev.soc) ** 2
                dep_seen.add(ev.idx)
        lmp = env._lmp(env.t_s / 3600.0) / 1000.0
        margin = sum((env.exec_prices[k] - lmp) * sum(env.agg_rates[k].values()) * 0.25
                     for k in range(env.n_agg))
        r = 2.0 * margin - 100.0 * unmet
        c = cost_signal(interval)
        if train:
            agent.store(s, raw, logp, r, c)
    if train:
        agent.update()
    return env.episode_metrics()


agent_cfg_cmax = 11.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True,
                    choices=["flat_ddpg", "ppo_lag", "cpo", "hrl"])
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--dataset", default="nrel")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n_ev", type=int, default=60)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ds = yaml.safe_load(open(ROOT / f"configs/dataset/{args.dataset}.yaml"))
    price = price_profile(ds.get("price_source", "pjm"))
    out = pathlib.Path(args.out or f"results/{args.method}_{args.network}_{args.dataset}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    np.random.seed(args.seed)

    if args.method == "flat_ddpg":
        agent = FlatDDPG(cfg)
    elif args.method == "ppo_lag":
        agent = PPOLagrangian(cfg); agent._cfg = cfg
    elif args.method == "cpo":
        agent = CPO(cfg); agent._cfg = cfg
    else:
        agent = HRLBaseline(cfg)

    fields = ["episode", "daily_cost_usd", "min_voltage_pu",
              "violation_rate_pct", "service_quality", "peak_mw", "wall_s"]
    with open(out / "train_log.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for ep in range(args.episodes):
            t0 = time.time()
            env = ChargingEnv(cfg, args.network, ds, n_ev=args.n_ev,
                              seed=args.seed * 10000 + ep,
                              scenario_mods={"disable_q_control": True})
            if args.method in ("flat_ddpg", "hrl"):
                m = agent.run_episode(env, price, LOAD_SHAPE, train=True)
            else:
                m = run_onpolicy_episode(agent, env, price, LOAD_SHAPE, train=True)
            row = {k: m.get(k) for k in fields if k in m}
            row.update(episode=ep, wall_s=round(time.time() - t0, 1))
            w.writerow(row); f.flush()
            if ep % 10 == 0:
                print(f"ep{ep:4d} cost=${m['daily_cost_usd']:.0f} "
                      f"vmin={m['min_voltage_pu']:.4f} viol={m['violation_rate_pct']:.1f}% "
                      f"SQ={m['service_quality']:.3f}", flush=True)
            if ep % 100 == 99:
                sp = str(out / f"ck_ep{ep}")
                agent.save(sp if args.method == "hrl" else sp + ".pt")
    agent.save(str(out / "final") if args.method == "hrl" else str(out / "final.pt"))
    print("done:", out)


if __name__ == "__main__":
    main()