#!/usr/bin/env python3
"""Train the nested framework.

Usage:
  python scripts/train.py --network ieee33 --dataset nrel --episodes 600 \
      --seed 0 --out results/nested_ieee33_nrel_s0
"""
import argparse, csv, json, pathlib, sys, time
sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1] / "src"))
import os
for v in ('OMP_NUM_THREADS','MKL_NUM_THREADS','OPENBLAS_NUM_THREADS','NUMEXPR_NUM_THREADS'):
    os.environ.setdefault(v, '1')
import torch; torch.set_num_threads(1)
import numpy as np
import yaml

from nflev.env.charging_env import ChargingEnv
from nflev.training.trainer import NestedTrainer

ROOT = pathlib.Path(__file__).resolve().parents[1]

# hourly LMP profiles ($/MWh) — replaced by real PJM/CAISO feeds on Day 3
PRICE_PROFILES = {
    "pjm": np.array([28,26,25,25,26,30,45,60,55,50,48,47,46,48,52,60,80,110,120,105,85,60,45,35], float),
}
LOAD_SHAPE = np.array([.55,.5,.48,.47,.48,.55,.65,.75,.72,.70,.68,.67,.66,.68,.72,.80,.92,1.0,1.0,.97,.88,.78,.68,.60])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--dataset", default="nrel")
    ap.add_argument("--episodes", type=int, default=600)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default=None)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--ablation", default="none",
                    choices=["none", "no_l1", "flat_timescale", "no_user_model"])
    ap.add_argument("--no_curriculum", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(ROOT / "configs/base.yaml"))
    ds = yaml.safe_load(open(ROOT / f"configs/dataset/{args.dataset}.yaml"))
    out = pathlib.Path(args.out or f"results/nested_{args.network}_{args.dataset}_s{args.seed}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "config_snapshot.yaml").write_text(yaml.dump(cfg))
    np.random.seed(args.seed)

    n_ev_max = 75

    def env_factory(stage, seed):
        n_ev = int(round(stage["penetration"] * n_ev_max))
        mods = {}
        if stage.get("episode_hours"):
            mods["episode_hours"] = stage["episode_hours"]
        if not stage.get("stochastic", True):
            mods["load_forecast_noise"] = 0.0
        if args.ablation == "no_user_model":
            mods["disable_behavior"] = True
        return ChargingEnv(cfg, args.network, ds, n_ev=n_ev, seed=seed,
                           scenario_mods=mods)

    abl = args.ablation if args.ablation != "no_user_model" else "none"
    import pathlib as _pl
    _pf = ROOT / f"configs/prices_{ds.get('price_source','pjm')}.yaml"
    if _pf.exists():
        PRICE_PROFILES[ds.get("price_source")] = np.array(
            yaml.safe_load(_pf.read_text())["hourly_lmp_usd_mwh"], float)
    trainer = NestedTrainer(cfg, env_factory, device=args.device, ablation=abl)
    if args.no_curriculum:
        trainer.curriculum.idx = 3
        trainer.curriculum.stages = [trainer.curriculum.stages[3]] * 4 + \
            [trainer.curriculum.stages[3]]
    price = PRICE_PROFILES[ds.get("price_source", "pjm")]

    log_path = out / "train_log.csv"
    fields = ["episode", "stage", "daily_cost_usd", "min_voltage_pu",
              "violation_rate_pct", "service_quality", "q_activation_freq",
              "curtailed_kwh", "peak_mw", "wall_s"]
    with open(log_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for ep in range(args.episodes):
            t0 = time.time()
            for ag in trainer.l2:
                ag.noise = max(0.05, 0.2 * (0.995 ** ep))
            eph = trainer.curriculum.stage.get("episode_hours", 24)
            lp = np.tile(LOAD_SHAPE, eph // 24 + 1)[:eph]
            pp_ = np.tile(price, eph // 24 + 1)[:eph]
            m = trainer.run_episode(seed=args.seed * 10000 + ep, train=True,
                                    price_profile=pp_, load_profile=lp)
            row = {k: m.get(k) for k in fields if k in m}
            row.update(episode=ep, stage=m["curriculum_stage"],
                       wall_s=round(time.time() - t0, 1))
            w.writerow(row); f.flush()
            if ep % 10 == 0 or m.get("advanced"):
                print(f"ep{ep:4d} stage{m['curriculum_stage']} "
                      f"cost=${m['daily_cost_usd']:.0f} vmin={m['min_voltage_pu']:.4f} "
                      f"Qact={m['q_activation_freq']:.2f} curt={m['curtailed_kwh']:.1f} "
                      f"{'-> ADVANCED' if m.get('advanced') else ''}", flush=True)
            if ep % 50 == 49:
                trainer.l1.save(out / f"l1_ep{ep}.pt")
                for k, ag in enumerate(trainer.l2):
                    ag.save(out / f"l2_{k}_ep{ep}.pt")

    trainer.l1.save(out / "l1_final.pt")
    for k, ag in enumerate(trainer.l2):
        ag.save(out / f"l2_{k}_final.pt")
    print(json.dumps({"done": True, "out": str(out)}))


if __name__ == "__main__":
    main()