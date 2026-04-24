#!/usr/bin/env bash
set -euo pipefail

python scripts/generate_dataset.py \
  --lens pm \
  --interval I1 \
  --output-dir data/processed \
  --n-y 8 \
  --n-omega 80 \
  --overwrite

python scripts/train_interval.py \
  --data data/processed/pm_I1_y8_w80.npy \
  --interval I1 \
  --output-dir runs/pm_I1_smoke \
  --epochs 5 \
  --batch-size 256 \
  --eval-every 1 \
  --save-every 5 \
  --hidden-dim 64 \
  --depth 3 \
  --fourier-feats 16

python scripts/evaluate_checkpoint.py \
  --checkpoint runs/pm_I1_smoke/best.pt \
  --lens pm \
  --interval I1 \
  --n-y 4 \
  --n-omega 40

python scripts/infer.py \
  --checkpoint runs/pm_I1_smoke/best.pt \
  --omega 1.0 \
  --y 0.8
