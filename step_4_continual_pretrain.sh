#!/usr/bin/env bash
set -e

# Adjust --nproc_per_node to your GPU count
torchrun --nproc_per_node=8 scripts/train_continual_pretrain.py       --config configs/continual_pretrain.yaml
