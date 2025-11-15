#!/usr/bin/env python
"""Launch a vLLM server for a K2 model.

This is a thin wrapper around `vllm.entrypoints.openai.api_server`.
It reads host/port from `configs/env.yaml` and accepts a `--model_path`
argument so you can easily swap between:

- ./models/k2_think_dpo
- ./models/k2_domain_instruct
- ./models/k2_student_7b
etc.
"""

import argparse
import subprocess
from typing import Dict

import yaml


def load_env_config(path: str) -> Dict:
    """Load shared environment configuration."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml",
    )
    parser.add_argument(
        "--model_path",
        default="./models/k2_think_dpo",
        help="Local path to a HF-compatible model directory.",
    )
    args = parser.parse_args()

    cfg = load_env_config(args.config)

    host = cfg["vllm"]["host"]
    port = cfg["vllm"]["port"]

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--host", host,
        "--port", str(port),
    ]
    print("[launch_teacher_vllm] Launching vLLM server:")
    print("  " + " ".join(cmd))
    subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()
