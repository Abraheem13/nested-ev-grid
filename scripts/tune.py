#!/usr/bin/env python3
"""Model selection on held-out training-year days (protocol: TUNING.md).

  python scripts/tune.py --jobs 4

For every candidate and seed: train the nested controller, calibrate its tariff
on training days, and evaluate it on the validation days under S3 and S7.
Resumable (finished steps are skipped). Writes artifacts/tuning/summary.csv and
artifacts/tuning/selected.json; configs/base.yaml is not modified. The final
policies are trained on all training-year days (data.val_days = 0).
"""
import argparse
import copy
import json
import os
import pathlib
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor

import yaml

ROOT = pathlib.Path(__file__).resolve().parents[1]
OUT = ROOT / "artifacts" / "tuning"
PY = sys.executable
CANDIDATES = {                      # id: (disaggregation, residual_penalty)  -- TUNING.md
    "A": ("llf", 0.0),
    "B": ("plan", 0.0),
    "C": ("plan", 0.25),
    "D": ("plan", 1.0),
}
SEEDS = (100, 101)
SCENARIOS = ("S3", "S7")
TIE = 0.002
VAL_DAYS = 30


def candidate_config(cid: str) -> pathlib.Path:
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    c = copy.deepcopy(cfg)
    c["level2"]["disaggregation"], c["level2"]["residual_penalty"] = CANDIDATES[cid]
    c["data"]["val_days"] = VAL_DAYS                       # held out from training for selection
    d = OUT / cid
    d.mkdir(parents=True, exist_ok=True)
    p = d / "config.yaml"
    p.write_text(yaml.safe_dump(c, sort_keys=False))
    return p


def run(cmd: list[str], log: pathlib.Path) -> None:
    with open(log, "a") as f:
        subprocess.run(cmd, cwd=ROOT, check=True, stdout=f, stderr=subprocess.STDOUT,
                       env={**os.environ, "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1"})


def job(cid: str, seed: int) -> str:
    cfgp = OUT / cid / "config.yaml"
    run_dir = OUT / cid / f"s{seed}"
    log = OUT / cid / f"s{seed}.log"
    if not (run_dir / "done.json").exists():
        run([PY, "scripts/train.py", "--method", "nested", "--seed", str(seed), "--config", str(cfgp),
             "--out", str(run_dir)], log)
    if not (run_dir / "price_calibration.json").exists():
        run([PY, "scripts/calibrate.py", "--method", "nested", "--checkpoint", str(run_dir / "model.pt"),
             "--config", str(cfgp)], log)
    n_val = yaml.safe_load(open(cfgp))["data"]["val_days"]
    for sc in SCENARIOS:
        out = run_dir / f"val_{sc}.csv"
        if not out.exists():
            run([PY, "scripts/evaluate.py", "--method", "nested", "--scenario", sc, "--split", "val",
                 "--episodes", str(n_val), "--checkpoint", str(run_dir / "model.pt"),
                 "--train-seed", str(seed), "--config", str(cfgp), "--out", str(out)], log)
    return f"{cid} s{seed}"


def summarize() -> dict:
    import pandas as pd
    rows = []
    for cid in CANDIDATES:
        for seed in SEEDS:
            for sc in SCENARIOS:
                f = OUT / cid / f"s{seed}" / f"val_{sc}.csv"
                d = pd.read_csv(f)
                rows.append({"candidate": cid, "seed": seed, "scenario": sc, "days": d.day.nunique(),
                             "cost": d.cost_eur.mean(), "unmet": d.unmet_kwh.mean(),
                             "curtailed": d.curtailed_kwh.mean(), "viol": d.violation_rate_pct.mean(),
                             "sq": d.service_quality.mean(), "retail": d.retail_price_paid.mean()})
    df = pd.DataFrame(rows)
    df["obj"] = df.cost + 2.0 * df.unmet + 0.5 * df.curtailed
    df.to_csv(OUT / "summary.csv", index=False)
    per = df.groupby(["candidate", "scenario"]).mean(numeric_only=True)
    J = {c: float(per.loc[(c, "S3"), "obj"] + per.loc[(c, "S7"), "obj"]) for c in CANDIDATES}
    ok = {c: bool(df[(df.candidate == c) & (df.scenario == "S3")].viol.max() == 0.0) for c in CANDIDATES}
    eligible = [c for c in CANDIDATES if ok[c]]
    best = min(eligible, key=J.get)
    for c in eligible:                                   # tie rule: first listed within 0.2 %
        if J[c] <= J[best] * (1 + TIE):
            best = c
            break
    sel = {"selected": best, "disaggregation": CANDIDATES[best][0], "residual_penalty": CANDIDATES[best][1],
           "J": J, "eligible": ok, "seeds": list(SEEDS), "protocol": "TUNING.md"}
    (OUT / "selected.json").write_text(json.dumps(sel, indent=2))
    print(per.round(3).to_string())
    print(json.dumps(sel, indent=2))
    return sel


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jobs", type=int, default=os.cpu_count())
    ap.add_argument("--summary-only", action="store_true")
    a = ap.parse_args()
    if not a.summary_only:
        for cid in CANDIDATES:
            candidate_config(cid)
        jobs = [(c, s) for s in SEEDS for c in CANDIDATES]
        with ProcessPoolExecutor(a.jobs) as ex:
            for r in ex.map(job, *zip(*jobs)):
                print("done", r, flush=True)
    summarize()


if __name__ == "__main__":
    main()
