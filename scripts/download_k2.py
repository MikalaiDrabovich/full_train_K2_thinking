#!/usr/bin/env python
"""Download Kimi K2 teacher models from Hugging Face.

This script reads `configs/env.yaml` and downloads all entries under
`teacher_models` into the local `models_dir`. It is intentionally simple:
- It does NOT manage versioning or partial checkpoints.
- It assumes you have enough disk space.
- It is meant to be run occasionally, not on every training launch.
"""

import argparse
import os
from typing import Dict

import yaml
from huggingface_hub import snapshot_download


def load_env_config(path: str) -> Dict:
    """Load the shared environment configuration YAML."""
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def download_all_teachers(cfg: Dict) -> None:
    """Download all teacher models listed in the env config.

    Expected config structure:

    teacher_models:
      k2_base: "moonshotai/Kimi-K2-Base"
      k2_instruct: "moonshotai/Kimi-K2-Instruct"
      k2_thinking: "moonshotai/Kimi-K2-Thinking"

    paths:
      models_dir: "./models"
    """
    hf_token = cfg.get("hf_token")
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    teacher_models = cfg.get("teacher_models", {})
    if not teacher_models:
        raise ValueError("No `teacher_models` section found in env.yaml")

    for name, repo_id in teacher_models.items():
        print(f"[download_k2] Downloading {name}: {repo_id}")
        out_dir = os.path.join(models_dir, name)

        # `snapshot_download` will reuse the cache if possible
        snapshot_download(
            repo_id,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=cfg.get("cache_dir"),
            resume_download=True,
        )
        print(f"[download_k2] Saved to {out_dir}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml (shared environment configuration).",
    )
    parser.add_argument(
        "--model-key",
        default=None,
        help=(
            "Optional key inside `teacher_models` if you only want to download "
            "a single entry (e.g. `k2_base`)."
        ),
    )
    args = parser.parse_args()

    cfg = load_env_config(args.config)

    if args.model_key:
        # Download just one specific model
        model_key = args.model_key
        teacher_models = cfg.get("teacher_models", {})
        if model_key not in teacher_models:
            raise KeyError(f"model-key={model_key!r} not found in `teacher_models`")
        sub_cfg = dict(cfg)
        sub_cfg["teacher_models"] = {model_key: teacher_models[model_key]}
        download_all_teachers(sub_cfg)
    else:
        # Download all teacher models
        download_all_teachers(cfg)


if __name__ == "__main__":
    main()
