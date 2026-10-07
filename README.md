# Nested Multi-Timescale Learning and Control for Voltage-Aware EV Charging Coordination

Code, data pipeline, trained policies and paper sources for the article by
Abraheem Rashid, Faisal Iradat, Waseem Iqbal and Yawar Abbas Bangash
(submitted to *IEEE Transactions on Transportation Electrification*).

The controller coordinates residential electric-vehicle charging on a
distribution feeder with three levels, one per timescale:

| Level | Timescale | Role |
|---|---|---|
| 1 | 1 h | PPO policy that sets the retail price corridor |
| 2 | 15 min | DDPG policy, shared by the aggregators, that corrects a deadline-aware cheapest-slot plan; least-laxity-first allocation keeps every dispatch within charger and transformer limits |
| 3 | 60 s | Non-learned voltage correction with charger reactive power bounded by the inverter rating (curtailment only when that headroom is exhausted) |

Drivers' price acceptance is part of the simulated environment, not a control level.

## Reproducing the paper

Requirements: Python 3.11 and the packages in `requirements.txt` (CPU only, no
GPU). A LaTeX installation with `pdflatex` is needed only to compile the PDF.

```bash
git clone https://github.com/Abraheem13/nested-ev-grid.git
cd nested-ev-grid
python -m venv .venv && source .venv/bin/activate    # Windows: .venv\Scripts\activate
pip install -r requirements.txt

python reproduce.py --from-results    # about 20 s: every table, figure and number from the stored results
python reproduce.py --jobs 8          # everything from the raw data (about 25 CPU-hours)
```

`python reproduce.py` runs the complete pipeline:

1. download the public data and verify every file by its SHA-256 checksum;
2. run the test suite;
3. model selection on 30 held-out 2023 days (`TUNING.md`);
4. train the nested controller, its ablations and the learned baselines
   (five seeds for the main comparison, three for ablations and generalization);
5. calibrate the tariff level of every learned policy on training days;
6. evaluate all methods on 50 held-out 2024 days;
7. record the 60-s operating profile of one representative test day, fixed by
   rule (`scripts/day_profile.py`; each run must reproduce its stored evaluation);
8. write every table, figure and number of the paper to `paper/generated/`,
   compile `paper/main.pdf` and check it (`scripts/check_paper.py`).

Every stage is resumable: finished jobs are skipped and interrupted training
runs continue from their last checkpoint (`--force` re-runs everything).
`python reproduce.py --quick` is a short functional check (one seed, 30
training episodes, three evaluation days).

## Reproducibility

* All numbers in the paper are LaTeX macros written by `nflev.eval.analysis`
  (`paper/generated/numbers.tex`); every qualitative statement is tagged and
  checked against the data (`paper/generated/claims.json`).
* Training and evaluation are seeded and single-threaded: retraining a policy
  reproduces its weights bit for bit, and re-evaluating a stored policy
  reproduces the stored results exactly.
* `requirements.txt` pins the package versions used for the paper;
  `artifacts/provenance.json` records the software environment of the run.

## Repository layout

| Path | Content |
|---|---|
| `reproduce.py` | One-command reproduction of the paper |
| `configs/base.yaml` | Every parameter of the study |
| `src/nflev/grid/` | IEEE 33- and 69-bus feeders (MATPOWER data) and radial AC power flow, validated against pandapower |
| `src/nflev/data/` | Day-ahead prices, household loads and charging sessions |
| `src/nflev/env/` | Quasi-static simulation at 60 s: driver acceptance model, Level-3 reactive correction, feasible allocation, planning prior |
| `src/nflev/training/` | Nested controller and training loop |
| `src/nflev/agents/` | PPO, DDPG, PPO-Lagrangian and CPO |
| `src/nflev/baselines/` | Uncoordinated, TOU timer, price-aware heuristic, MPC LP-OPF, flat DDPG, safe RL, hierarchical RL |
| `src/nflev/eval/` | Paired evaluation, statistics, tables, figures, number macros and claims |
| `scripts/` | Training, calibration, evaluation, model selection and paper checks |
| `tests/` | Unit and regression tests |
| `artifacts/` | Trained policies, training logs, evaluation results, operating profiles, model selection, provenance |
| `paper/` | LaTeX sources and the compiled paper |

## Data

All data are public and are downloaded automatically:

* Netherlands day-ahead prices from the ENTSO-E Transparency Platform and
  Pecan Street residential load profiles, as distributed with `ev2gym==2.0.0`;
* ACN-Data charging sessions (Caltech, JPL), as distributed with `sustaingym==0.1.7`;
* IEEE 33- and 69-bus feeders from MATPOWER (`data/networks/`, see `SOURCE.md`).

Household load data were provided by Pecan Street Inc. (Dataport).

## Previous version

The code base used for the earlier draft of the paper is available under the
git tag `v2`.
