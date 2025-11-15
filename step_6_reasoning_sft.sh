#!/usr/bin/env bash
set -e

torchrun --nproc_per_node=8 scripts/train_sft_reasoning.py       --config configs/sft_reasoning.yaml
