#!/usr/bin/env python3
"""Tariff calibration of a trained policy (training data only).

A learned policy sets retail prices; the revenue-neutral multiplier keeps the
*training-time* mean retail price near lambda_ref, but the final deterministic
policy can end up above or below it. This script finds, by bisection, the
constant offset added to every execution price such that the deterministic
policy's mean retail price paid (total revenue / delivered energy) on
`--days` days of the TRAINING year equals lambda_ref, and writes it to
<run dir>/price_calibration.json, which evaluation applies.

  python scripts/calibrate.py --method nested \
      --checkpoint artifacts/runs/nested_residential_ieee33_none_s0/model.pt
"""
import argparse
import json
import os
import pathlib
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import numpy as np  # noqa: E402
import torch        # noqa: E402
import yaml         # noqa: E402

torch.set_num_threads(1)
CALIB_SEED = 500_000            # disjoint from training (10 000 (seed + 1) + episode) and test seeds


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, help="nested | flat_ddpg | ppo_lag | cpo | hrl")
    ap.add_argument("--checkpoint", required=True)
    ap.add_argument("--fleet", default="residential")
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--days", type=int, default=10)
    ap.add_argument("--iters", type=int, default=7)
    ap.add_argument("--bracket", type=float, default=0.10, help="largest |offset| searched (EUR/kWh)")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    from nflev.env.charging_env import ChargingEnv
    from nflev.env.episode import make_episode
    from nflev.eval.methods import build
    from nflev.eval.runner import run_policy_episode

    ckpt = pathlib.Path(a.checkpoint)
    policy, q_on, env_cfg = build(a.method, cfg, cfg["aggregators"]["n"], ckpt, calibrated=False)
    env = ChargingEnv(env_cfg, a.network, q_control=q_on)
    lo, hi = cfg["curriculum"]["stages"][-1]["n_ev_range"]
    rng = np.random.default_rng(CALIB_SEED)
    specs = [make_episode(env_cfg, a.fleet, "train", int(rng.integers(lo, hi + 1)), seed=CALIB_SEED + j)
             for j in range(a.days)]
    ref = cfg["behavior"]["lambda_ref"]

    def retail(offset: float) -> float:
        policy.price_offset = offset
        rev = deliv = 0.0
        for spec in specs:
            m = run_policy_episode(env, policy, spec)
            rev += m["retail_price_paid"] * m["energy_delivered_kwh"]
            deliv += m["energy_delivered_kwh"]
        return rev / deliv

    p0 = retail(0.0)
    a_lo, a_hi = (0.0, a.bracket) if p0 < ref else (-a.bracket, 0.0)
    off, p = 0.0, p0
    if abs(p0 - ref) > 5e-4:
        for _ in range(a.iters):          # retail price paid increases with the offset
            off = 0.5 * (a_lo + a_hi)
            p = retail(off)
            if p < ref:
                a_lo = off
            else:
                a_hi = off
            if abs(p - ref) < 5e-4:
                break
    out = {"price_offset": off, "retail_uncalibrated": p0, "retail_calibrated": p,
           "lambda_ref": ref, "days": a.days, "split": "train", "seed": CALIB_SEED}
    (ckpt.parent / "price_calibration.json").write_text(json.dumps(out, indent=2))
    print(json.dumps(out))


if __name__ == "__main__":
    main()
