# Model-selection protocol (fixed before any candidate was trained)

The v3.0 results (commit `1ae0ca5`) identified two sources of the nested
controller's nominal-load cost premium over the price-aware heuristic: the
least-laxity-first execution of the plan and the learned residual. This round
tests two remedies. Selection uses **validation days only**; the 2024 test
set is evaluated once, with the selected configuration, after selection.

## Data
* Validation: `data.val_days = 30` days of the training year 2023, drawn by a
  fixed permutation (`VAL_PERMUTATION_SEED`), removed from training for every
  method. Calibration (`scripts/calibrate.py`) uses training days only.
* Conditions: S3 (240 vehicles) and S7 (360 vehicles, inverter rating
  halved) on all 30 validation days.

## Candidates (nested controller; all other settings as in `configs/base.yaml`)
| id | `level2.disaggregation` | `level2.residual_penalty` (beta) |
|---|---|---|
| A | llf  | 0.0  (v3.0 method) |
| B | plan | 0.0 |
| C | plan | 0.25 |
| D | plan | 1.0 |

`plan`: urgent vehicles first, then the vehicles whose cheapest-slot plan
selects the current interval, then the rest, each group by increasing laxity.
Each candidate is trained with two seeds (100, 101; disjoint from the test
seeds 0-4), tariff-calibrated, and evaluated on the validation days.

## Criterion
For each candidate, averaged over its two seeds and the 30 validation days:

    J = [cost + 2 * unmet + 0.5 * curtailed]_S3 + [cost + 2 * unmet + 0.5 * curtailed]_S7

(EUR and kWh per day; the weights are the Level-2 reward weights). Candidates
with any S3 validation violation are excluded. The lowest J is selected; if
two candidates are within 0.2 % of each other, the one listed first wins.

## Outputs
`scripts/tune.py` runs the protocol and writes
`artifacts/tuning/summary.csv` and `artifacts/tuning/selected.json`. The
selected values are then written into `configs/base.yaml`, and the full
campaign (`reproduce.py`) is re-run, with five training seeds for the main
comparison.

## Result (run after the protocol above was committed)
Validation means over the two seeds and 30 days (`artifacts/tuning/summary.csv`):

| id | S3 cost | S3 unmet | S7 cost | S7 unmet | S7 curtailed | J |
|---|---|---|---|---|---|---|
| A | 722.8 | 24.0 | 1109.9 | 35.4 | 45.2 | **1974.1** |
| B | 719.8 | 33.3 | 1107.7 | 56.4 | 59.7 | 2036.7 |
| C | 713.6 | 29.8 | 1097.9 | 49.4 | 95.4 | 2017.5 |
| D | 716.9 | 45.4 | 1101.8 | 71.7 | 124.0 | 2114.8 |

No candidate had an S3 validation violation. **A (least-laxity-first, no residual
penalty) has the lowest J and is retained**: plan-priority allocation lowers
the S3 energy cost by up to 1.3 % but leaves 24-89 % more energy undelivered at
S3 and 39-103 % more at S7. Observation (not used for selection): the
calibrated retail price of B-D on the validation days was 0.207-0.214
EUR/kWh against 0.199 for A, which lowers acceptance and contributes to their
higher unmet energy.

Because A is the existing configuration, the final policies are those of the
main campaign, trained on all 2023 days (`data.val_days = 0`); seeds 3 and 4
were added to the main comparison.
