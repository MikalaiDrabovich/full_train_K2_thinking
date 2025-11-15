#!/usr/bin/env python
import argparse
import os
import yaml
from huggingface_hub import snapshot_download

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    hf_token = cfg.get("hf_token")
    models_dir = cfg["paths"]["models_dir"]
    os.makedirs(models_dir, exist_ok=True)

    for name, repo_id in cfg["teacher_models"].items():
        print(f"Downloading {name}: {repo_id}")
        out_dir = os.path.join(models_dir, name)
        snapshot_download(
            repo_id,
            local_dir=out_dir,
            local_dir_use_symlinks=False,
            token=hf_token,
            cache_dir=cfg.get("cache_dir"),
            resume_download=True,
        )
        print(f"Saved to {out_dir}")

if __name__ == "__main__":
    main()
