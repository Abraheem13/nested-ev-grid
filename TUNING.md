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
