# CIFAR-10 Vision Training

This directory contains from-scratch CIFAR-10 classifiers for learning the standard PyTorch training workflow.

## Structure

- `train.py`: command-line entry point, argument parsing, run setup.
- `data.py`: HuggingFace CIFAR-10 loading, train/validation split, augmentation, dataloaders.
- `engine.py`: train/eval loops and checkpoint saving.
- `logging_utils.py`: TensorBoard and W&B logging examples.
- `models/cnn/model.py`: explicit small ResNet-style CNN.
- `models/vit/model.py`: explicit tiny ViT with local patch embedding, attention, MLP, and encoder blocks.
- `configs/vision/cnn.yaml`, `configs/vision/vit.yaml`: experiment configs.
- `scripts/train_cnn.sh`, `scripts/train_vit.sh`: runnable training presets.

## Quick Runs

```bash
# CNN
vision/scripts/train_cnn.sh

# ViT
vision/scripts/train_vit.sh

# Fast smoke test
vision/scripts/train_cnn.sh --debug
```

## Config-First Training

Training is driven by Hydra/OmegaConf YAML configs:

```bash
.venv/bin/python -m vision.train
.venv/bin/python -m vision.train --config-name vit
```

Use Hydra-style command-line overrides:

```bash
vision/scripts/train_cnn.sh experiment.run_name=cnn-test experiment.seed=123 logging.use_wandb=false
```

```bash
vision/scripts/train_cnn.sh \
  train.lr=1.0e-3 \
  train.batch_size=256 \
  logging.wandb_group=cifar10-ablation
```

Use `debug=true` for a tiny offline smoke run:

```bash
vision/scripts/train_cnn.sh debug=true logging.use_wandb=false
```

Keep secrets and machine-specific settings in `.env`, such as `WANDB_API_KEY`, `HF_TOKEN`, `DEVICE`, `CUDA_VISIBLE_DEVICES`, and output/cache roots. Keep experiment hyperparameters in YAML or Hydra overrides.

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

The code reads `.env` through `python-dotenv`. Useful environment fields include:

- `WANDB_API_KEY`, `WANDB_PROJECT`, `WANDB_ENTITY`, `WANDB_MODE`
- `OUTPUT_DIR`, `CHECKPOINT_DIR`, `LOG_DIR`, `CACHE_DIR`
- `DEVICE`, `CUDA_VISIBLE_DEVICES`

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
  train.epochs=20 \
  train.lr=3.0e-4
```

Use `--config-name vit` to switch to the transformer. Use `logging.use_wandb=false` or `logging.use_tensorboard=false` to disable either logger.
