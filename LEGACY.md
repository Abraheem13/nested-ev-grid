# Legacy v2 material (removed from the working tree)

The files below belonged to the superseded v2 code base (the version behind the
originally drafted paper). The v3 pipeline (`reproduce.py`) never used them, so
they were removed to keep the repository unambiguous. **Nothing is lost**: all
of them are preserved in the git history at commit

    0153f1ed9087b98c73de6501df805823c4ad7ffd

on this branch (also on GitHub).

## Restore

Restore everything:

    git checkout 0153f1ed9087b98c73de6501df805823c4ad7ffd -- results figs configs/dataset configs/prices_caiso.yaml \
        scripts/evaluate_parallel.py scripts/evaluate_v2.py scripts/make_figures.py \
        scripts/run_ablations.sh scripts/run_campaign.py scripts/run_grid.sh \
        scripts/train_baseline.py scripts/verify_claims.py \
        src/nflev/agents/projection.py src/nflev/data/loaders.py src/nflev/env/ev_fleet.py \
        src/nflev/env/network.py src/nflev/eval/stats.py src/nflev/utils \
        nohup.out logs_abl1.out logs_abl2.out logs_abl3.out logs_abl4.out logs_acn.out logs_cpo.out logs_eval_bl.out logs_flat_ddpg.out logs_hrl.out logs_n_69.out logs_n_acn.out logs_n_elaad.out logs_n_s0.out logs_n_s1.out logs_n_s2.out logs_ppo_lag.out logs_s0.out logs_v2s0.out logs_v3s0.out 

or any single path, e.g. `git checkout 0153f1ed9087b98c73de6501df805823c4ad7ffd -- results/`.

## What was removed and what replaced it

| Removed | Content | Replaced by (v3) |
|---|---|---|
| `results/` (138 files) | v2 evaluation CSVs, tables, claims report, run logs | `artifacts/` written by `reproduce.py` |
| `figs/` | v2 figures | `paper/generated/` |
| `logs_*.out`, `nohup.out` | v2 console logs | `artifacts/logs/` |
| `configs/dataset/`, `configs/prices_caiso.yaml` | v2 hand-typed fleet and price parameters | real data via `nflev.data` (ENTSO-E, Pecan Street, ACN-Data) |
| `scripts/evaluate_parallel.py`, `evaluate_v2.py`, `run_campaign.py`, `run_ablations.sh`, `run_grid.sh`, `train_baseline.py` | v2 run scripts | `reproduce.py`, `scripts/train.py`, `scripts/evaluate.py`, `scripts/calibrate.py` |
| `scripts/make_figures.py`, `scripts/verify_claims.py` | v2 figure/claim scripts | `nflev.eval.analysis`, `scripts/check_paper.py` |
| `src/nflev/agents/projection.py` | v2 projection | `nflev.env.allocation` |
| `src/nflev/data/loaders.py` | v2 dataset-config builder | `nflev.data.{prices,loads,fleets,sources}` |
| `src/nflev/env/ev_fleet.py` | v2 fleet generator | `nflev.data.fleets` |
| `src/nflev/env/network.py` | v2 pandapower network wrapper | `nflev.grid.{network,powerflow}` |
| `src/nflev/eval/stats.py` | v2 statistics | `nflev.eval.analysis` |
| `src/nflev/utils/` | empty package | -- |
