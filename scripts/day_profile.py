#!/usr/bin/env python3
"""Operating profile of one representative test day (60-s resolution).

  python scripts/day_profile.py

The day is fixed by a rule, not chosen by eye: the S3 test day on which
uncoordinated charging has the median daily cost (lower median of the 50
days). On that day the script runs uncoordinated charging with and without
Level 3, the price-aware heuristic with Level 3 and the nested controller of
training seed 0, records every 60-s step, and checks that each run reproduces
the stored evaluation result of that day exactly. Writes artifacts/profiles/.
"""
import os
import pathlib
import sys

for v in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(v, "1")
ROOT = pathlib.Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

import pandas as pd  # noqa: E402
import torch         # noqa: E402
import yaml          # noqa: E402

torch.set_num_threads(1)
ART = ROOT / "artifacts"
SCENARIO = "S3"
SEED = 0
METHODS = {           # method -> (checkpoint, stored evaluation file)
    "uncoordinated": (None, "main__uncoordinated__S3__residential__ieee33__test__s-1.csv"),
    "price_aware+L3": (None, "main__price_aware+L3__S3__residential__ieee33__test__s-1.csv"),
    "nested": (f"runs/nested_residential_ieee33_none_s{SEED}/model.pt",
               f"main__nested__S3__residential__ieee33__test__s{SEED}.csv"),
    "uncoordinated+L3": (None, "main__uncoordinated+L3__S3__residential__ieee33__test__s-1.csv"),
}
CHECK = ["cost_eur", "violation_rate_pct", "min_voltage_pu", "curtailed_kwh", "service_quality"]


def representative_day() -> int:
    d = pd.read_csv(ART / "eval" / METHODS["uncoordinated"][1]).sort_values(["cost_eur", "episode"])
    return int(d.episode.iloc[(len(d) - 1) // 2])


def main():
    from nflev.env.charging_env import ChargingEnv
    from nflev.env.episode import make_episode
    from nflev.eval.methods import build
    from nflev.eval.runner import run_policy_episode
    cfg = yaml.safe_load(open(ROOT / "configs" / "base.yaml"))
    ev = cfg["evaluation"]
    scen = dict(ev["scenarios"][SCENARIO])
    n_ev = scen.pop("n_ev")
    j = representative_day()
    out = ART / "profiles"
    out.mkdir(parents=True, exist_ok=True)
    for method, (ckpt, stored) in METHODS.items():
        policy, q_on, env_cfg = build(method, cfg, cfg["aggregators"]["n"], ROOT / "artifacts" / ckpt if ckpt else None)
        env = ChargingEnv(env_cfg, "ieee33", q_control=q_on)
        spec = make_episode(env_cfg, "residential", "test", n_ev, seed=ev["seed_base"] + j, scenario=dict(scen),
                            eval_index=j)
        env.trace = []
        m = run_policy_episode(env, policy, spec)
        ref = pd.read_csv(ART / "eval" / stored).set_index("episode").loc[j]
        bad = [k for k in CHECK if abs(float(m[k]) - float(ref[k])) > 1e-9]
        if bad:
            raise SystemExit(f"{method}: profile run differs from the stored evaluation in {bad}")
        tr = pd.DataFrame(env.trace)
        tr.insert(0, "step", range(len(tr)))
        tr["method"], tr["episode"], tr["day"] = method, j, m["day"]
        tr.to_csv(out / f"{SCENARIO}__{method}.csv", index=False)
        print(f"{method}: day {m['day']} (episode {j}), cost {m['cost_eur']:.1f}, "
              f"violations {m['violation_rate_pct']:.2f}%  [matches stored evaluation]")


if __name__ == "__main__":
    main()
