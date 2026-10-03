#!/usr/bin/env python3
"""Train one learned method.

  python scripts/train.py --method nested --fleet residential --network ieee33 \
      --seed 0 --ablation none --out artifacts/runs/nested_residential_ieee33_none_s0
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

import torch  # noqa: E402
import yaml   # noqa: E402

torch.set_num_threads(1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True, choices=["nested", "flat_ddpg", "ppo_lag", "cpo", "hrl"])
    ap.add_argument("--fleet", default="residential")
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--ablation", default="none")
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    out = pathlib.Path(a.out)
    if a.method == "nested":
        from nflev.training.trainer import train_nested
        info = train_nested(cfg, a.fleet, a.network, a.seed, a.ablation, out, a.episodes)
    elif a.method == "flat_ddpg":
        from nflev.baselines.flat_ddpg import train_flat_ddpg
        info = train_flat_ddpg(cfg, a.fleet, a.network, a.seed, out, a.episodes)
    elif a.method == "hrl":
        from nflev.baselines.hrl import train_hrl
        info = train_hrl(cfg, a.fleet, a.network, a.seed, out, a.episodes)
    else:
        from nflev.baselines.safe_rl import train_safe_rl
        info = train_safe_rl(cfg, a.method, a.fleet, a.network, a.seed, out, a.episodes)
    (out / "done.json").write_text(json.dumps({**vars(a), **info}, indent=2))


if __name__ == "__main__":
    main()
