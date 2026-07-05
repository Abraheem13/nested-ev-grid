#!/usr/bin/env bash
# Complete experiment grid. Split across nodes by argument:
#   ./scripts/run_grid.sh node1   # nested: seeds x datasets + 69-bus
#   ./scripts/run_grid.sh node2   # baselines + ablations
# Everything sequential per node; logs to logs_grid_*.out.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true
ROLE=${1:?usage: run_grid.sh node1|node2}
EP=600

if [ "$ROLE" = "node1" ]; then
  # nested framework: 3 seeds x nrel, 1 seed x acn + elaad, 69-bus generalization
  for S in 0 1 2; do
    python scripts/train.py --episodes $EP --seed $S --dataset nrel \
        --out results/nested_ieee33_nrel_s$S > logs_grid_nested_s$S.out 2>&1
  done
  for D in acn elaad; do
    python scripts/train.py --episodes $EP --seed 0 --dataset $D \
        --out results/nested_ieee33_${D}_s0 > logs_grid_nested_$D.out 2>&1
  done
  python scripts/train.py --episodes $EP --seed 0 --network ieee69 \
      --out results/nested_ieee69_nrel_s0 > logs_grid_nested_69.out 2>&1
  # evaluation sweeps
  python scripts/evaluate.py --methods uncoordinated tou milp nested \
      --scenarios S1 S2 S3 S4 S5 S6_deadlock S7_q_exhaustion --episodes 50 \
      --checkpoint results/nested_ieee33_nrel_s0 \
      --out results/eval_ieee33_nrel.csv > logs_grid_eval.out 2>&1
  python scripts/evaluate.py --methods uncoordinated tou nested \
      --scenarios S3 --episodes 50 --network ieee69 \
      --checkpoint results/nested_ieee69_nrel_s0 \
      --out results/eval_ieee69_nrel.csv > logs_grid_eval69.out 2>&1

elif [ "$ROLE" = "node2" ]; then
  for M in flat_ddpg ppo_lag cpo hrl; do
    python scripts/train_baseline.py --method $M --episodes $EP --seed 0 \
        --out results/${M}_ieee33_nrel_s0 > logs_grid_$M.out 2>&1
  done
  python scripts/evaluate.py --methods flat_ddpg ppo_lag cpo hrl \
      --scenarios S1 S2 S3 S4 S5 --episodes 50 \
      --checkpoint results/nested_ieee33_nrel_s0 \
      --out results/eval_learners_ieee33.csv > logs_grid_eval_learners.out 2>&1
  bash scripts/run_ablations.sh > logs_grid_ablations.out 2>&1
fi
echo "grid role $ROLE complete"