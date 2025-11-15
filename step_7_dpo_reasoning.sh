#!/usr/bin/env bash
set -e

torchrun --nproc_per_node=8 scripts/run_dpo_reasoning.py       --config configs/dpo_reasoning.yaml
