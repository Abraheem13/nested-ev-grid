#!/usr/bin/env python3
"""One-click reproduction of every number, table and figure in the paper.

    python reproduce.py                 # everything (data -> tests -> model selection -> train -> calibrate -> eval -> paper)
    python reproduce.py --jobs 8        # parallel workers (default: CPU count)
    python reproduce.py --from-results  # only regenerate tables/figures/numbers from artifacts/
    python reproduce.py --quick         # short functional check: 1 seed, 30 training episodes, 3 eval days

Every stage is resumable: a job whose output already exists is skipped
(--force re-runs). Outputs:
    artifacts/runs/<run>/model.pt, train_log.csv   trained policies
    artifacts/eval/<job>.csv                       one row per evaluation episode
    artifacts/provenance.json                      git commit, config hash, versions
    paper/generated/                               tables (*.tex), numbers.tex, figures (*.pdf)
    paper/main.pdf                                 if a LaTeX engine is installed
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import pathlib
import platform
import shutil
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parent
ART = ROOT / "artifacts"
PY = sys.executable
SEEDS = [0, 1, 2, 3, 4]          # main comparison (nested and learned baselines)
N_AUX_SEEDS = 3                  # ablations and generalization use the first three
ABLATIONS = ["no_prior", "no_l1", "flat_timescale", "no_behavior", "no_l3", "no_curriculum", "no_guard",
             "proportional"]
LEARNED_BASELINES = ["flat_ddpg", "ppo_lag", "cpo", "hrl"]
RULES = ["uncoordinated", "tou", "price_aware", "lp_opf"]
MAIN_SCEN = ["S1", "S2", "S3", "S4", "S5"]


def run_dir(method, fleet, network, ablation, seed):
    return ART / "runs" / f"{method}_{fleet}_{network}_{ablation}_s{seed}"


def train_jobs(seeds, episodes):
    jobs = []

    aux = seeds[:N_AUX_SEEDS]

    def add(method, fleet="residential", network="ieee33", ablation="none", seeds=seeds):
        for s in seeds:
            out = run_dir(method, fleet, network, ablation, s)
            cmd = [PY, "scripts/train.py", "--method", method, "--fleet", fleet, "--network", network,
                   "--seed", str(s), "--ablation", ablation, "--out", str(out)]
            if episodes:
                cmd += ["--episodes", str(episodes)]
            jobs.append((out.name, cmd, out / "done.json"))

    add("nested")
    for ab in ABLATIONS:
        add("nested", ablation=ab, seeds=aux)
    add("nested", network="ieee69", seeds=aux)
    add("nested", fleet="acn_caltech", seeds=aux)
    add("nested", fleet="acn_jpl", seeds=aux)
    for m in LEARNED_BASELINES:
        add(m)
    return jobs


def calib_jobs(seeds, episodes):
    """Tariff calibration of every learned policy (training days only)."""
    jobs = []
    for name, cmd, sentinel in train_jobs(seeds, episodes):
        out = sentinel.parent
        arg = dict(zip(cmd[2::2], cmd[3::2]))
        method = arg["--method"]
        if method == "nested" and arg["--ablation"] == "no_l3":
            method = "nested-noL3"                     # evaluated without Level 3
        c = [PY, "scripts/calibrate.py", "--method", method, "--checkpoint", str(out / "model.pt"),
             "--fleet", arg["--fleet"], "--network", arg["--network"]]
        jobs.append((f"calib__{name}", c, out / "price_calibration.json"))
    return jobs


def eval_jobs(seeds, episodes):
    jobs = []

    def add(method, scenario, fleet="residential", network="ieee33", split="test", ckpt=None,
            seed=-1, variant="main", overrides=()):
        tag = f"{variant}__{method}__{scenario}__{fleet}__{network}__{split}__s{seed}"
        out = ART / "eval" / f"{tag}.csv"
        cmd = [PY, "scripts/evaluate.py", "--method", method, "--scenario", scenario, "--fleet", fleet,
               "--network", network, "--split", split, "--train-seed", str(seed),
               "--variant", variant, "--out", str(out)]
        if ckpt:
            cmd += ["--checkpoint", str(ckpt)]
        if episodes:
            cmd += ["--episodes", str(episodes)]
        if overrides:
            cmd += ["--override", *overrides]
        jobs.append((tag, cmd, out))

    def nested_ckpt(s, fleet="residential", network="ieee33", ablation="none"):
        return run_dir("nested", fleet, network, ablation, s) / "model.pt"

    for sc in MAIN_SCEN + ["S6", "S7"]:
        for r in RULES:
            add(r, sc)
            add(r + "+L3", sc)
        for s in seeds:
            add("nested", sc, ckpt=nested_ckpt(s), seed=s)
            for m in LEARNED_BASELINES:
                ck = run_dir(m, "residential", "ieee33", "none", s) / "model.pt"
                add(m, sc, ckpt=ck, seed=s)
                add(m + "+L3", sc, ckpt=ck, seed=s)
    for sc in ("S3", "S4"):                                    # base load alone (reference)
        add("noev", sc, variant="reference")
    for sc in ("S3", "S7"):                                    # what the learned levels add
        add("plan+L3", sc, variant="decomp")
        for s in seeds:
            for m in ("nested-flatprice", "nested-planprice"):
                add(m, sc, ckpt=nested_ckpt(s), seed=s, variant="decomp")
    aux = seeds[:N_AUX_SEEDS]
    for s in aux:                                              # ablations on S3 and S5
        add("nested-noL3", "S3", ckpt=nested_ckpt(s), seed=s, variant="ablation")
        for ab in ABLATIONS:
            for sc in ("S3", "S5"):
                add("nested-noL3" if ab == "no_l3" else "nested", sc,
                    ckpt=nested_ckpt(s, ablation=ab), seed=s, variant=f"abl_{ab}")
    for fleet, network in (("residential", "ieee69"), ("acn_caltech", "ieee33"), ("acn_jpl", "ieee33")):
        for r in RULES:
            add(r, "S3", fleet, network, variant="general")
            add(r + "+L3", "S3", fleet, network, variant="general")
        for s in aux:
            add("nested", "S3", fleet, network, ckpt=nested_ckpt(s, fleet, network), seed=s, variant="general")
    add("uncoordinated", "S3", split="alt", variant="regime")  # pre-crisis price regime (2019)
    for r in RULES:
        add(r + "+L3", "S3", split="alt", variant="regime")
    for s in seeds:
        add("nested", "S3", split="alt", ckpt=nested_ckpt(s), seed=s, variant="regime")
    for s in seeds:                                            # Level-3 sensitivity
        for name, ov in (("delta0.0005", "voltage.correction_margin=0.0005"),
                         ("delta0.002", "voltage.correction_margin=0.002"),
                         ("S10", "reactive_power.s_rated_kva=10.0"),
                         ("S14", "reactive_power.s_rated_kva=14.0")):
            add("nested", "S3", ckpt=nested_ckpt(s), seed=s, variant=f"sens_{name}", overrides=(ov,))
    return jobs


def run_job(job, force):
    name, cmd, sentinel = job
    if sentinel.exists() and not force:
        return name, True, 0.0, ""
    log = ART / "logs" / f"{name}.log"
    log.parent.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    with open(log, "w") as f:
        f.write("$ " + " ".join(map(str, cmd)) + "\n")
        f.flush()
        rc = subprocess.run(cmd, cwd=ROOT, stdout=f, stderr=subprocess.STDOUT).returncode
    ok = rc == 0 and sentinel.exists()
    return name, ok, time.time() - t0, "" if ok else str(log)


def run_all(jobs, n_workers, force, label):
    todo = [j for j in jobs if force or not j[2].exists()]
    print(f"== {label}: {len(jobs)} jobs, {len(todo)} to run, {n_workers} workers", flush=True)
    failed = []
    with cf.ProcessPoolExecutor(n_workers) as ex:
        futs = {ex.submit(run_job, j, force): j for j in todo}
        for i, fut in enumerate(cf.as_completed(futs), 1):
            name, ok, dt, log = fut.result()
            print(f"  [{i}/{len(todo)}] {'ok  ' if ok else 'FAIL'} {name} ({dt / 60:.1f} min)", flush=True)
            if not ok:
                failed.append((name, log))
    if failed:
        for n, lg in failed:
            print(f"  FAILED {n}: see {lg}")
        raise SystemExit(f"{label}: {len(failed)} job(s) failed")


def provenance():
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        except OSError:
            return "unknown"
    import numpy, scipy, torch, pandas  # noqa: E401
    cfg = (ROOT / "configs" / "base.yaml").read_bytes()
    return {"git_commit": git("rev-parse", "HEAD"), "git_dirty": bool(git("status", "--porcelain")),
            "config_sha256": hashlib.sha256(cfg).hexdigest(), "python": platform.python_version(),
            "numpy": numpy.__version__, "scipy": scipy.__version__, "torch": torch.__version__,
            "pandas": pandas.__version__, "platform": platform.platform(),
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}


def compile_paper():
    """Compile paper/main.tex (three passes resolve references) and run the
    static and log checks of scripts/check_paper.py."""
    engine = shutil.which("pdflatex")
    if engine is None:
        print("== paper: no LaTeX engine found; running static checks only")
        subprocess.run([PY, "scripts/check_paper.py"], cwd=ROOT, check=True)
        return
    for _ in range(3):
        subprocess.run([engine, "-interaction=nonstopmode", "-halt-on-error", "main.tex"],
                       cwd=ROOT / "paper", check=True, stdout=subprocess.DEVNULL)
    subprocess.run([PY, "scripts/check_paper.py", "--log"], cwd=ROOT, check=True)
    print("== paper: paper/main.pdf (all checks passed)")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--jobs", type=int, default=os.cpu_count() or 2)
    ap.add_argument("--from-results", action="store_true")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--force", action="store_true")
    ap.add_argument("--skip-tests", action="store_true")
    a = ap.parse_args()
    sys.path.insert(0, str(ROOT / "src"))
    global ART
    if a.quick:
        ART = ROOT / "artifacts_quick"
    seeds = [0] if a.quick else SEEDS
    tr_eps = 30 if a.quick else None
    ev_eps = 3 if a.quick else None
    if not a.from_results:
        from nflev.data.sources import ensure_data
        ensure_data()
        if not a.skip_tests:
            print("== tests", flush=True)
            subprocess.run([PY, "-m", "pytest", "-q", "tests"], cwd=ROOT, check=True)
        ART.mkdir(parents=True, exist_ok=True)
        import yaml
        from nflev.env.calibration import write_report
        write_report(yaml.safe_load(open(ROOT / "configs" / "base.yaml")), ART / "calibration.json")
        if not a.quick:              # model selection on held-out 2023 days (TUNING.md); resumable
            print("== model selection (scripts/tune.py)", flush=True)
            subprocess.run([PY, "scripts/tune.py", "--jobs", str(a.jobs)], cwd=ROOT, check=True)
        run_all(train_jobs(seeds, tr_eps), a.jobs, a.force, "train")
        run_all(calib_jobs(seeds, tr_eps), a.jobs, a.force, "calibrate tariffs")
        run_all(eval_jobs(seeds, ev_eps), a.jobs, a.force, "evaluate")
        (ART / "provenance.json").write_text(json.dumps(provenance(), indent=2))
    from nflev.eval.analysis import build_all
    build_all(ART, ROOT / "paper" / "generated")
    if not a.quick:                  # the quick check lacks data for several claims
        compile_paper()


if __name__ == "__main__":
    main()
