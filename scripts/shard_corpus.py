#!/usr/bin/env python
"""Shard large JSONL corpora into smaller files.

This is a convenience utility for cases where you want to:
- parallelize processing over multiple workers;
- or simply avoid single multi‑GB JSONL files.

The script reads an input JSONL file and writes `shard_00000.jsonl`,
`shard_00001.jsonl`, ... into a `shards/` directory next to it.
"""

import argparse
import os
from typing import Dict

import yaml


def shard_jsonl(in_path: str, out_dir: str, shard_size: int = 100_000) -> None:
    """Shard a JSONL file into multiple shards with up to `shard_size` lines.

    The function is deliberately simple:
    - does not validate JSON lines;
    - assumes input is UTF‑8 encoded;
    - preserves the original lines as‑is.
    """
    os.makedirs(out_dir, exist_ok=True)
    shard_idx = 0
    count = 0
    out = None

    with open(in_path, "r", encoding="utf-8") as f:
        for line in f:
            if out is None or count >= shard_size:
                if out:
                    out.close()
                shard_path = os.path.join(out_dir, f"shard_{shard_idx:05d}.jsonl")
                print(f"[shard_corpus] Opening {shard_path}")
                out = open(shard_path, "w", encoding="utf-8")
                shard_idx += 1
                count = 0
            out.write(line)
            count += 1

    if out:
        out.close()
    print(f"[shard_corpus] Done. Created {shard_idx} shards.")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml (for `paths` section).",
    )
    parser.add_argument(
        "--corpus-key",
        default="domain_corpus",
        help=(
            "Key inside `paths` that contains the JSONL corpus path "
            "(e.g. domain_corpus or reasoning_corpus)."
        ),
    )
    parser.add_argument(
        "--shard-size",
        type=int,
        default=100_000,
        help="Maximum number of lines per shard.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    in_path = cfg["paths"][args.corpus_key]
    out_dir = os.path.join(os.path.dirname(in_path), "shards")
    shard_jsonl(in_path, out_dir, shard_size=args.shard_size)


if __name__ == "__main__":
    main()
