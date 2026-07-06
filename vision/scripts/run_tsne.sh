#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/../.."

.venv/bin/python -m vision.tsne_embeddings \
  --models cnn vit \
  --cnn-checkpoint checkpoints/vision/cnn-cifar10-20260706-132602/best.pt \
  --vit-checkpoint checkpoints/vision/vit-cifar10-20260706-132622/best.pt \
  --split test \
  --num-samples 2000 \
  --batch-size 256 \
  --num-workers 4 \
  "$@"
