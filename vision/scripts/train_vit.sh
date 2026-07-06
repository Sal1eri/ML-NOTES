#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m vision.train \
  --model vit \
  --epochs 30 \
  --batch-size 128 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --num-workers 4 \
  --log-every 20 \
  --hist-every 200 \
  "$@"

