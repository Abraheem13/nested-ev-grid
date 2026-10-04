#!/usr/bin/env python3
"""Evaluate one method on one scenario (50 held-out days by default).

  python scripts/evaluate.py --method price_aware+L3 --scenario S3 --out artifacts/eval/x.csv
  python scripts/evaluate.py --method nested --scenario S3 \
      --checkpoint artifacts/runs/nested_residential_ieee33_none_s0/model.pt --out ...
"""
import argparse
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
    ap.add_argument("--method", required=True)
    ap.add_argument("--scenario", required=True)
    ap.add_argument("--fleet", default="residential")
    ap.add_argument("--network", default="ieee33")
    ap.add_argument("--split", default="test", choices=["test", "alt", "val"])
    ap.add_argument("--episodes", type=int, default=None)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--train-seed", type=int, default=-1)
    ap.add_argument("--variant", default="main")
    ap.add_argument("--override", nargs="*", default=[], help="dot.path=value config overrides")
    ap.add_argument("--config", default=str(ROOT / "configs" / "base.yaml"))
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    cfg = yaml.safe_load(open(a.config))
    for ov in a.override:
        path, val = ov.split("=", 1)
        node = cfg
        keys = path.split(".")
        for k in keys[:-1]:
            node = node[k]
        if keys[-1] not in node:
            raise SystemExit(f"unknown config key {path}")
        node[keys[-1]] = yaml.safe_load(val)
    from nflev.eval.evaluate import evaluate, write_rows
    rows = evaluate(a.method, cfg, a.scenario, a.fleet, a.network, a.split, a.episodes,
                    pathlib.Path(a.checkpoint) if a.checkpoint else None,
                    extra={"train_seed": a.train_seed, "variant": a.variant})
    write_rows(rows, pathlib.Path(a.out))


if __name__ == "__main__":
    main()
