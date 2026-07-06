#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m vision.nearest_neighbors \
  --model vit \
  --checkpoint checkpoints/vision/vit-cifar10-20260706-132622/best.pt \
  --num-samples 2000 \
  --num-query-classes 5 \
  --neighbors 5 \
  --batch-size 256 \
  --num-workers 4 \
  "$@"
