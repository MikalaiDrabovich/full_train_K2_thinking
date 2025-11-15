#!/usr/bin/env bash
set -e

torchrun --nproc_per_node=8 scripts/train_sft_domain.py       --config configs/sft_domain.yaml
