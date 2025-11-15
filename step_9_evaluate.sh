#!/usr/bin/env bash
set -e

python scripts/evaluate_model.py       --config configs/env.yaml       --model_path ./models/k2_student_7b
