#!/usr/bin/env bash
# Full ablation suite (Table V). Run AFTER the complete framework (nested_v3_s0)
# has trained. Sequential; ~5 x 3.5 h training + evaluation.
set -e
cd "$(dirname "$0")/.."
source .venv/bin/activate 2>/dev/null || true

EP=600
DS=nrel
NET=ieee33

# --- variants requiring retraining -------------------------------------
python scripts/train.py --episodes $EP --seed 0 --ablation no_l1 \
    --out results/abl_no_l1            > logs_abl_no_l1.out 2>&1
python scripts/train.py --episodes $EP --seed 0 --ablation flat_timescale \
    --out results/abl_flat_timescale   > logs_abl_flat.out 2>&1
python scripts/train.py --episodes $EP --seed 0 --ablation no_user_model \
    --out results/abl_no_user_model    > logs_abl_nouser.out 2>&1
python scripts/train.py --episodes $EP --seed 0 --no_curriculum \
    --out results/abl_no_curriculum    > logs_abl_nocurr.out 2>&1

# --- evaluation of every variant on S3 (50 episodes) --------------------
# complete framework
python scripts/evaluate.py --methods nested --scenarios S3 --episodes 50 \
    --checkpoint results/nested_v3_s0 --out results/abl_complete.csv
# no Level 3 Q control: complete agents evaluated with Q disabled
python - << 'EOF'
import subprocess, sys, yaml, pathlib
# evaluate.py has no q-disable flag for nested; do it via scenario mod clone
cfg = yaml.safe_load(open("configs/base.yaml"))
cfg["evaluation"]["scenarios"]["S3_noq"] = dict(cfg["evaluation"]["scenarios"]["S3"])
cfg["evaluation"]["scenarios"]["S3_noq"]["disable_q_control"] = True
pathlib.Path("configs/base.yaml").write_text(yaml.dump(cfg, sort_keys=False))
EOF
python scripts/evaluate.py --methods nested --scenarios S3_noq --episodes 50 \
    --checkpoint results/nested_v3_s0 --out results/abl_no_l3.csv
# retrained variants
for V in no_l1 flat_timescale no_user_model no_curriculum; do
  python scripts/evaluate.py --methods nested --scenarios S3 --episodes 50 \
      --checkpoint results/abl_$V --out results/abl_$V.csv
done
echo "Ablation suite complete. CSVs in results/abl_*.csv"