# Nested multi-timescale learning for EV charging coordination

Code, data pipeline and paper sources for *"Charge Fully, Stay Safe: Nested
Multi-Timescale Learning for EV Charging Coordination"* (submitted to IEEE
Transactions on Sustainable Energy).

**One command regenerates every number, table and figure in the paper:**

```bash
pip install -r requirements.txt
python reproduce.py --jobs 8        # data -> tests -> training -> evaluation -> paper
```

`reproduce.py` is resumable (finished jobs are skipped, interrupted training
runs resume from their last checkpoint). `python reproduce.py --from-results`
rebuilds only the tables, figures and `paper/generated/numbers.tex` from the
stored evaluation logs; `python reproduce.py --quick` is a smoke run (one seed,
30 training episodes, three evaluation days). If `pdflatex` is installed the
paper is compiled and checked automatically (`scripts/check_paper.py`).

## What is in the box

| Path | Content |
|---|---|
| `src/nflev/grid/` | MATPOWER IEEE 33/69-bus feeders, radial AC power flow (backward/forward sweep, validated against pandapower Newton-Raphson to 1e-8 p.u.) |
| `src/nflev/data/` | ENTSO-E NL day-ahead prices, Pecan Street household loads, ACN-Data sessions (downloaded from pinned PyPI wheels, SHA-256 verified) |
| `src/nflev/env/` | 60-s quasi-static simulation: price acceptance (L3a), reactive correction (L3b), feasibility layer (projection, least-laxity-first allocation), planning prior |
| `src/nflev/training/` | Nested controller: L1 PPO price corridor, L2 plan-residual DDPG dispatch, rewards, curriculum, resumable training |
| `src/nflev/baselines/` | Uncoordinated, TOU timer, price-aware heuristic, perfect-foresight MPC LP-OPF, flat DDPG, PPO-Lagrangian, CPO, hierarchical RL |
| `src/nflev/eval/` | Paired evaluation on held-out days, statistics, tables, figures, number macros, data-checked claims |
| `configs/base.yaml` | Every parameter of the study (read by the code; nothing decorative) |
| `paper/` | LaTeX sources; `paper/generated/` is written by `reproduce.py`; `REFERENCES_VERIFICATION.md` documents how every reference was verified |
| `artifacts/` | Trained policies, training logs, per-episode evaluation logs, provenance |
| `tests/` | Power flow, data integrity, feasibility, energy accounting, Level 3, training-pipeline tests |

## Data

All data are public and fetched automatically by `nflev.data.sources.ensure_data()`:

* Netherlands day-ahead prices (ENTSO-E Transparency Platform) and Pecan Street
  residential load profiles, as bundled in `ev2gym==2.0.0`;
* ACN-Data charging sessions (Caltech, JPL), as bundled in `sustaingym==0.1.7`;
* IEEE 33- and 69-bus feeders from MATPOWER (vendored in `data/networks/`, see `SOURCE.md`).

Household load data were provided by Pecan Street Inc. (Dataport).

## Legacy material

The superseded v2 code, results and logs were removed from the working tree;
they remain in the git history. `LEGACY.md` lists every removed path, what
replaced it, and the one-line command that restores it.
