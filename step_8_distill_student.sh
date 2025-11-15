#!/usr/bin/env bash
set -e

torchrun --nproc_per_node=8 scripts/distill_to_student.py       --config configs/distill_student.yaml
