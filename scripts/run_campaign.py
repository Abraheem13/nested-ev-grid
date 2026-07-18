#!/usr/bin/env python3
"""End-to-end experimental campaign for the nested-EV-grid manuscript.

This orchestrator runs EVERY experiment the paper reports, in dependency
order, with hard integrity gates in between. Its purpose is to make the
published tables mechanically reproducible from one command:

    Phase 1  TRAIN   nested (33/NREL, 69/NREL, 33/ACN, 33/ElaadNL),
                     4 training-time ablations, 4 learned baselines.
    Phase 2  GATE    refuse to proceed unless every nested/ablation run
                     completed its full episode budget AND the curriculum
                     reached the final stage (this is the check that the
                     previous campaign silently failed).
    Phase 3  EVAL    frozen-policy evaluation, 50 episodes, seeds 7000-7049:
                     - main grid: 8 methods x S1-S5           (Table IV)
                     - stress:    nested on S6, S7            (Sec. stress)
                     - ablations: 5 rows on S3                (Table V)
                     - 69-bus:    uncoordinated/TOU/nested S3 (Table VII)
                     - datasets:  nested on ACN / ElaadNL S3  (Table VII)
                     - sensitivity: delta / S-rating sweeps   (Table VIII)
    Phase 4  STATS   mean +/- 95% CI, Welch t-tests, Holm correction
                     (delegates to nflev.eval.stats).
    Phase 5  VERIFY  scripts/verify_claims.py checks every headline claim of
                     the manuscript against the fresh CSVs and emits the
                     LaTeX table rows + a PASS/FAIL claims report.

Design principles
-----------------
* Subprocess isolation: one crashed run never corrupts another; every run has
  its own log file under results/campaign/logs/.
* Resumability: a run whose sentinel artifact already exists is skipped, so
  the campaign can be re-launched after interruption (``--force`` re-runs).
* Provenance: git commit, dirty flag, config hash, package versions, seeds,
  and per-run wall time are recorded in results/campaign/manifest.json.
* Determinism: training seeds are fixed per run; evaluation seeds are the
  fixed protocol seeds 7000..7049.

Typical usage
-------------
    # inspect the plan without executing anything
    python scripts/run_campaign.py --dry-run

    # full campaign, 4 concurrent workers, resumable
    nohup python scripts/run_campaign.py --jobs 4 > campaign.out 2>&1 &

    # re-run a single phase
    python scripts/run_campaign.py --phase eval --jobs 4
"""
from __future__ import annotations

import argparse
import concurrent.futures as cf
import dataclasses
import hashlib
import json
import pathlib
import shlex
import subprocess
import sys
import time

ROOT = pathlib.Path(__file__).resolve().parents[1]
PY = sys.executable
RESULTS = ROOT / "results"
CAMP = RESULTS / "campaign"
LOGS = CAMP / "logs"
EVAL = RESULTS / "eval"
TABLES = RESULTS / "tables"

TRAIN_EPISODES = 600
EVAL_EPISODES = 50
SEED_BASE = 7000

MAIN_METHODS = ["uncoordinated", "tou", "milp", "nested",
                "flat_ddpg", "ppo_lag", "cpo", "hrl"]
MAIN_SCENARIOS = ["S1", "S2", "S3", "S4", "S5"]


# --------------------------------------------------------------------------- #
# job model
# --------------------------------------------------------------------------- #
@dataclasses.dataclass(frozen=True)
class Job:
    name: str                 # unique id, also the log-file stem
    cmd: list[str]            # argv
    sentinel: pathlib.Path    # exists  =>  job considered complete
    phase: str                # train | eval | stats
    deps: tuple[str, ...] = ()

    def done(self) -> bool:
        return self.sentinel.exists()


def sh(cmd: list[str]) -> str:
    return " ".join(shlex.quote(c) for c in cmd)


def run_job(job: Job, force: bool) -> tuple[str, bool, float]:
    """Execute one job; returns (name, ok, wall_seconds)."""
    if job.done() and not force:
        return job.name, True, 0.0
    LOGS.mkdir(parents=True, exist_ok=True)
    log = LOGS / f"{job.name}.log"
    t0 = time.time()
    with open(log, "w") as lf:
        lf.write(f"# {sh(job.cmd)}\n")
        lf.flush()
        rc = subprocess.run(job.cmd, cwd=ROOT, stdout=lf,
                            stderr=subprocess.STDOUT).returncode
    ok = (rc == 0) and job.done()
    if rc == 0 and not job.done():
        with open(log, "a") as lf:
            lf.write(f"\n# ERROR: exit 0 but sentinel missing: {job.sentinel}\n")
    return job.name, ok, time.time() - t0


# --------------------------------------------------------------------------- #
# campaign definition
# --------------------------------------------------------------------------- #
def training_jobs() -> list[Job]:
    jobs: list[Job] = []

    def train_nested(tag, network, dataset, extra=()):
        out = RESULTS / f"nested_{tag}"
        jobs.append(Job(
            name=f"train_nested_{tag}",
            cmd=[PY, "scripts/train.py", "--network", network,
                 "--dataset", dataset, "--episodes", str(TRAIN_EPISODES),
                 "--seed", "0", "--out", str(out), *extra],
            sentinel=out / "l1_final.pt", phase="train"))

    # main checkpoints (one per network/dataset the paper reports)
    train_nested("ieee33_nrel", "ieee33", "nrel")
    train_nested("ieee69_nrel", "ieee69", "nrel")
    train_nested("ieee33_acn", "ieee33", "acn")
    train_nested("ieee33_elaad", "ieee33", "elaad")

    # training-time ablations (Table V). "No Level 3" and the sensitivity
    # sweeps reuse the main checkpoint at eval time and need no training run.
    train_nested("abl_no_l1", "ieee33", "nrel", ("--ablation", "no_l1"))
    train_nested("abl_flat_timescale", "ieee33", "nrel",
                 ("--ablation", "flat_timescale"))
    train_nested("abl_no_user_model", "ieee33", "nrel",
                 ("--ablation", "no_user_model"))
    train_nested("abl_no_curriculum", "ieee33", "nrel", ("--no_curriculum",))

    # learned baselines - output dirs must match evaluate_v2's lookup pattern
    for meth in ["flat_ddpg", "ppo_lag", "cpo", "hrl"]:
        out = RESULTS / f"{meth}_ieee33_nrel_s0"
        sentinel = out / ("final_high.pt" if meth == "hrl" else "final.pt")
        jobs.append(Job(
            name=f"train_{meth}",
            cmd=[PY, "scripts/train_baseline.py", "--method", meth,
                 "--episodes", str(TRAIN_EPISODES), "--network", "ieee33",
                 "--dataset", "nrel", "--seed", "0", "--out", str(out)],
            sentinel=sentinel, phase="train"))
    return jobs


def eval_jobs() -> list[Job]:
    jobs: list[Job] = []
    ck_main = str(RESULTS / "nested_ieee33_nrel")

    def ev(name, out_csv, extra, deps=()):
        jobs.append(Job(
            name=f"eval_{name}",
            cmd=[PY, "scripts/evaluate_v2.py", "--episodes",
                 str(EVAL_EPISODES), "--seed_base", str(SEED_BASE),
                 "--out", str(out_csv), *extra],
            sentinel=out_csv, phase="eval", deps=deps))

    # -- main grid (Table IV) + statistical table (Table VI) -----------------
    ev("main_grid", EVAL / "main_ieee33_nrel.csv",
       ["--methods", *MAIN_METHODS, "--scenarios", *MAIN_SCENARIOS,
        "--network", "ieee33", "--dataset", "nrel",
        "--checkpoint", ck_main, "--baseline_ckpts", str(RESULTS)],
       deps=("train_nested_ieee33_nrel", "train_flat_ddpg",
             "train_ppo_lag", "train_cpo", "train_hrl"))

    # -- stress scenarios (S6 deadlock, S7 capacity exhaustion) --------------
    ev("stress", EVAL / "stress_ieee33_nrel.csv",
       ["--methods", "nested", "--scenarios", "S6_deadlock",
        "S7_q_exhaustion", "--network", "ieee33", "--dataset", "nrel",
        "--checkpoint", ck_main],
       deps=("train_nested_ieee33_nrel",))

    # -- ablations on S3 (Table V) -------------------------------------------
    abl = [
        ("abl_no_l1",           RESULTS / "nested_abl_no_l1",           ["--ablation", "no_l1"]),
        ("abl_flat_timescale",  RESULTS / "nested_abl_flat_timescale",  ["--ablation", "flat_timescale"]),
        ("abl_no_user_model",   RESULTS / "nested_abl_no_user_model",   ["--env_mods", "disable_behavior=true"]),
        ("abl_no_curriculum",   RESULTS / "nested_abl_no_curriculum",   []),
        # eval-time ablation: full checkpoint, reactive layer switched off
        ("abl_no_l3",           RESULTS / "nested_ieee33_nrel",         ["--env_mods", "disable_q_control=true"]),
    ]
    for tag, ck, extra in abl:
        dep = ("train_nested_ieee33_nrel" if tag == "abl_no_l3"
               else f"train_nested_{tag}")
        ev(tag, EVAL / f"{tag}.csv",
           ["--methods", "nested", "--scenarios", "S3",
            "--network", "ieee33", "--dataset", "nrel",
            "--checkpoint", str(ck), *extra],
           deps=(dep,))

    # -- cross-network: IEEE 69-bus (Table VII, upper block) -----------------
    ev("ieee69", EVAL / "ieee69_nrel.csv",
       ["--methods", "uncoordinated", "tou", "nested", "--scenarios", "S3",
        "--network", "ieee69", "--dataset", "nrel",
        "--checkpoint", str(RESULTS / "nested_ieee69_nrel")],
       deps=("train_nested_ieee69_nrel",))

    # -- cross-dataset: ACN, ElaadNL (Table VII, lower block) ----------------
    for dsname in ["acn", "elaad"]:
        ev(f"dataset_{dsname}", EVAL / f"dataset_{dsname}.csv",
           ["--methods", "nested", "--scenarios", "S3",
            "--network", "ieee33", "--dataset", dsname,
            "--checkpoint", str(RESULTS / f"nested_ieee33_{dsname}")],
           deps=(f"train_nested_ieee33_{dsname}",))

    # -- hyperparameter sensitivity on S3 (Table VIII) -----------------------
    sens = [
        ("delta_0p0005", "voltage.correction_margin=0.0005"),
        ("delta_0p002",  "voltage.correction_margin=0.002"),
        ("srated_10",    "reactive_power.s_rated_kva=10.0"),
        ("srated_14",    "reactive_power.s_rated_kva=14.0"),
    ]
    for tag, override in sens:
        ev(f"sens_{tag}", EVAL / f"sens_{tag}.csv",
           ["--methods", "nested", "--scenarios", "S3",
            "--network", "ieee33", "--dataset", "nrel",
            "--checkpoint", ck_main, "--cfg_override", override],
           deps=("train_nested_ieee33_nrel",))
    return jobs


def stats_jobs() -> list[Job]:
    TABLES.mkdir(parents=True, exist_ok=True)
    return [Job(
        name="stats_main",
        cmd=[PY, "-m", "nflev.eval.stats",
             str(EVAL / "main_ieee33_nrel.csv"), "--out", str(TABLES)],
        sentinel=TABLES / ".stats_done", phase="stats",
        deps=("eval_main_grid",))]


# --------------------------------------------------------------------------- #
# curriculum gate
# --------------------------------------------------------------------------- #
def curriculum_gate(strict: bool) -> list[str]:
    """Verify every nested training run finished its budget and reached the
    final curriculum stage. Returns the list of failures."""
    import csv as _csv
    import yaml as _yaml
    n_stages = len(_yaml.safe_load(open(ROOT / "configs/base.yaml"))
                   ["curriculum"]["stages"])
    failures = []
    for d in sorted(RESULTS.glob("nested_*")):
        log = d / "train_log.csv"
        if not log.exists():
            continue
        rows = list(_csv.DictReader(open(log)))
        n_ep = len(rows)
        max_stage = max((int(r["stage"]) for r in rows), default=-1)
        no_curr = "no_curriculum" in d.name
        ok_ep = n_ep >= TRAIN_EPISODES
        ok_stage = no_curr or (max_stage >= n_stages - 1)
        status = "OK" if (ok_ep and ok_stage) else "FAIL"
        print(f"  [{status}] {d.name}: episodes={n_ep}/{TRAIN_EPISODES}, "
              f"max_stage={max_stage}/{n_stages - 1}"
              f"{' (curriculum disabled)' if no_curr else ''}")
        if status == "FAIL":
            failures.append(d.name)
    if failures and strict:
        print("\nGATE FAILED - the following runs are incomplete and their "
              "checkpoints MUST NOT be used for the paper:")
        for f in failures:
            print(f"  - {f}")
    return failures


# --------------------------------------------------------------------------- #
# provenance
# --------------------------------------------------------------------------- #
def provenance() -> dict:
    def git(*a):
        try:
            return subprocess.run(["git", *a], cwd=ROOT, capture_output=True,
                                  text=True).stdout.strip()
        except Exception:
            return "unknown"
    cfg = (ROOT / "configs/base.yaml").read_bytes()
    return {
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "git_commit": git("rev-parse", "HEAD"),
        "git_dirty": bool(git("status", "--porcelain")),
        "base_config_sha256": hashlib.sha256(cfg).hexdigest(),
        "python": sys.version.split()[0],
        "train_episodes": TRAIN_EPISODES,
        "eval_episodes": EVAL_EPISODES,
        "eval_seeds": f"{SEED_BASE}..{SEED_BASE + EVAL_EPISODES - 1}",
    }


# --------------------------------------------------------------------------- #
# scheduler
# --------------------------------------------------------------------------- #
def execute(jobs: list[Job], jobs_n: int, force: bool) -> bool:
    """Dependency-aware parallel execution. Returns True if all jobs passed."""
    done = {j.name for j in jobs if j.done() and not force}
    failed: set[str] = set()
    pending = [j for j in jobs if j.name not in done]
    results: dict[str, float] = {}

    with cf.ProcessPoolExecutor(max_workers=jobs_n) as pool:
        futures: dict[cf.Future, str] = {}
        while pending or futures:
            ready = [j for j in pending
                     if all(d in done for d in j.deps)
                     and not any(d in failed for d in j.deps)]
            blocked_by_failure = [j for j in pending
                                  if any(d in failed for d in j.deps)]
            for j in blocked_by_failure:
                print(f"  [SKIP] {j.name} (dependency failed)")
                failed.add(j.name)
                pending.remove(j)
            for j in ready:
                if len(futures) >= jobs_n:
                    break
                print(f"  [RUN ] {j.name}")
                futures[pool.submit(run_job, j, force)] = j.name
                pending.remove(j)
            if not futures:
                if pending:
                    for j in pending:
                        print(f"  [DEAD] {j.name} unschedulable "
                              f"(deps: {j.deps})")
                    return False
                break
            fut = next(cf.as_completed(list(futures)))
            name = futures.pop(fut)
            _, ok, wall = fut.result()
            results[name] = wall
            if ok:
                done.add(name)
                print(f"  [DONE] {name} ({wall / 60:.1f} min)")
            else:
                failed.add(name)
                print(f"  [FAIL] {name} - see {LOGS / (name + '.log')}")

    manifest = CAMP / "manifest.json"
    CAMP.mkdir(parents=True, exist_ok=True)
    prev = json.loads(manifest.read_text()) if manifest.exists() else {}
    prev.setdefault("provenance", provenance())
    prev.setdefault("wall_seconds", {}).update(results)
    prev["failed"] = sorted(failed)
    manifest.write_text(json.dumps(prev, indent=2))
    return not failed


def main() -> None:
    ap = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--phase", choices=["all", "train", "gate", "eval",
                                        "stats", "verify"], default="all")
    ap.add_argument("--jobs", type=int, default=2,
                    help="concurrent worker processes")
    ap.add_argument("--dry-run", action="store_true",
                    help="print the plan and exit")
    ap.add_argument("--force", action="store_true",
                    help="re-run jobs whose sentinels already exist")
    ap.add_argument("--no-gate", action="store_true",
                    help="UNSAFE: evaluate even if training gate fails")
    args = ap.parse_args()

    train = training_jobs()
    evals = eval_jobs()
    stats = stats_jobs()

    if args.dry_run:
        print(json.dumps(provenance(), indent=2))
        for phase, js in [("TRAIN", train), ("EVAL", evals), ("STATS", stats)]:
            print(f"\n== {phase} ({len(js)} jobs) ==")
            for j in js:
                mark = "done" if j.done() else "todo"
                print(f"  [{mark}] {j.name}\n         {sh(j.cmd)}")
        return

    ok = True
    if args.phase in ("all", "train"):
        print(f"\n===== PHASE 1: TRAIN ({len(train)} jobs) =====")
        ok = execute(train, args.jobs, args.force)

    if args.phase in ("all", "gate", "eval"):
        print("\n===== PHASE 2: CURRICULUM / COMPLETION GATE =====")
        failures = curriculum_gate(strict=True)
        if failures and not args.no_gate:
            sys.exit("Aborting before evaluation. Fix training first "
                     "(or pass --no-gate at your own risk).")

    if ok and args.phase in ("all", "eval"):
        print(f"\n===== PHASE 3: EVAL ({len(evals)} jobs) =====")
        ok = execute(train + evals, args.jobs, args.force)

    if ok and args.phase in ("all", "stats"):
        print("\n===== PHASE 4: STATS =====")
        ok = execute(train + evals + stats, args.jobs, args.force)
        (TABLES / ".stats_done").touch()

    if ok and args.phase in ("all", "verify"):
        print("\n===== PHASE 5: VERIFY CLAIMS =====")
        rc = subprocess.run([PY, "scripts/verify_claims.py"], cwd=ROOT).returncode
        ok = ok and rc == 0

    print("\nCampaign", "COMPLETE - all jobs passed." if ok
          else "INCOMPLETE - inspect results/campaign/logs/.")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()