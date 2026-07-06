# CIFAR-10 Vision Training

This directory contains from-scratch CIFAR-10 classifiers for learning the standard PyTorch training workflow.

## Structure

- `train.py`: command-line entry point, argument parsing, run setup.
- `data.py`: HuggingFace CIFAR-10 loading, train/validation split, augmentation, dataloaders.
- `engine.py`: train/eval loops and checkpoint saving.
- `logging_utils.py`: TensorBoard and W&B logging examples.
- `models/cnn/model.py`: explicit small ResNet-style CNN.
- `models/vit/model.py`: explicit tiny ViT with local patch embedding, attention, MLP, and encoder blocks.
- `scripts/train_cnn.sh`, `scripts/train_vit.sh`: runnable training presets.

## Quick Runs

```bash
# CNN
vision/scripts/train_cnn.sh

# ViT
vision/scripts/train_vit.sh

# Fast smoke test
vision/scripts/train_cnn.sh --epochs 1 --limit-train 512 --limit-val 128 --limit-test 128 --wandb-mode offline
```

## Embedding t-SNE

After training, visualize the embedding before the final classifier:

```bash
vision/scripts/run_tsne.sh
```

This reads the default dated checkpoints:

- `checkpoints/vision/cnn-cifar10-20260706-132602/best.pt`
- `checkpoints/vision/vit-cifar10-20260706-132622/best.pt`

Outputs are written to `outputs/vision/<run-name>/`:

- `cnn_tsne.png`
- `vit_tsne.png`
- `cnn_tsne_embeddings.npz`
- `vit_tsne_embeddings.npz`

Use different checkpoints like this:

```bash
vision/scripts/run_tsne.sh \
  --cnn-checkpoint checkpoints/vision/my-cnn/best.pt \
  --vit-checkpoint checkpoints/vision/my-vit/best.pt
```

## Embedding Nearest Neighbors

Show whether nearby embeddings correspond to visually or semantically similar images:

```bash
vision/scripts/run_neighbors.sh
```

Default behavior uses the dated CNN checkpoint, samples 2000 test images as the search space, randomly selects 5 classes, then shows one query image and its 5 nearest neighbors by cosine similarity.

For ViT:

```bash
vision/scripts/run_neighbors_vit.sh
```

Outputs are written to `outputs/vision/<run-name>/`:

- `cnn_nearest_neighbors.png`
- `cnn_nearest_neighbors.npz`

You can choose classes explicitly:

```bash
vision/scripts/run_neighbors.sh --classes airplane cat ship truck frog
```

The code reads `.env` through `python-dotenv`. Useful fields include:

- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`
- `OUTPUT_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`, `CACHE_DIR`
- `SEED`, `DEVICE`, `BATCH_SIZE`

## Logging

TensorBoard logs include config text, environment text, sample images, model graph, scalar curves, confusion matrices, and parameter/gradient histograms.

```bash
tensorboard --logdir logs/vision/<run-name>/tensorboard
```

W&B logs include config, tags, scalar curves, sample images, prediction tables, confusion matrices, watched gradients/weights, histograms, and model checkpoint artifacts.

For offline runs:

```bash
wandb sync logs/vision/<run-name>/wandb/offline-run-*
```

## Common Arguments

```bash
.venv/bin/python -m vision.train \
  --model cnn \
  --epochs 20 \
  --batch-size 128 \
  --lr 3e-4 \
  --weight-decay 0.05 \
  --use-wandb \
  --use-tensorboard
```

Use `--model vit` to switch to the transformer. Use `--no-use-wandb` or `--no-use-tensorboard` to disable either logger.
