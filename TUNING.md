# Model selection

Two design alternatives for Level 2 were screened before the final evaluation:
a plan-priority allocation (serve each vehicle's planned slots first) and a
penalty on the learned residual. The protocol below was fixed before any
candidate was trained. Selection uses validation days only; the 2024 test set
is evaluated once, with the selected configuration.

## Data
* Validation: 30 days of the training year 2023, drawn by a fixed permutation
  (`VAL_PERMUTATION_SEED` in `src/nflev/env/episode.py`) and removed from
  training for every candidate. Tariff calibration uses training days only.
* Conditions: S3 (240 vehicles) and S7 (360 vehicles, inverter rating halved)
  on all 30 validation days.

## Candidates
All other settings as in `configs/base.yaml`.

| id | `level2.disaggregation` | `level2.residual_penalty` |
|---|---|---|
| A | `llf` (least-laxity-first) | 0.0 |
| B | `plan` | 0.0 |
| C | `plan` | 0.25 |
| D | `plan` | 1.0 |

`plan` serves urgent vehicles first, then the vehicles whose cheapest-slot plan
selects the current interval, then the rest, each group by increasing laxity.
Each candidate is trained with two seeds (100, 101; disjoint from the test
seeds 0-4), tariff-calibrated and evaluated on the validation days.

## Criterion
For each candidate, averaged over its two seeds and the 30 validation days:

    J = [cost + 2 * unmet + 0.5 * curtailed]_S3 + [cost + 2 * unmet + 0.5 * curtailed]_S7

(EUR and kWh per day; the weights are the Level-2 reward weights). Candidates
with any S3 validation violation are excluded. The lowest J is selected; if
two candidates are within 0.2 % of each other, the one listed first wins.

## Result
Validation means over the two seeds and 30 days (`artifacts/tuning/summary.csv`):

| id | S3 cost | S3 unmet | S7 cost | S7 unmet | S7 curtailed | J |
|---|---|---|---|---|---|---|
| A | 722.8 | 24.0 | 1109.9 | 35.4 | 45.2 | **1974.1** |
| B | 719.8 | 33.3 | 1107.7 | 56.4 | 59.7 | 2036.7 |
| C | 713.6 | 29.8 | 1097.9 | 49.4 | 95.4 | 2017.5 |
| D | 716.9 | 45.4 | 1101.8 | 71.7 | 124.0 | 2114.8 |

No candidate had an S3 validation violation. Candidate A (least-laxity-first,
no residual penalty) has the lowest J and is used in the paper: plan-priority
allocation lowers the S3 energy cost by up to 1.3 % but leaves 24-89 % more
energy undelivered at S3 and 39-103 % more at S7. The calibrated retail price
of B-D on the validation days was 0.207-0.214 EUR/kWh against 0.199 EUR/kWh
for A, which lowers acceptance and contributes to their higher unmet energy
(an observation, not part of the criterion).

The final policies are trained on all 2023 days (`data.val_days = 0` in
`configs/base.yaml`).

## Reproduction
`python scripts/tune.py --jobs 4` runs the protocol (resumable) and writes
`artifacts/tuning/summary.csv` and `artifacts/tuning/selected.json`;
`reproduce.py` runs it as one of its stages.
