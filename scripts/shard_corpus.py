#!/usr/bin/env python
import argparse
import json
import os
import yaml

def shard_jsonl(in_path, out_dir, shard_size=100000):
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
                print(f"Opening {shard_path}")
                out = open(shard_path, "w", encoding="utf-8")
                shard_idx += 1
                count = 0
            out.write(line)
            count += 1

    if out:
        out.close()
    print(f"Done. {shard_idx} shards.")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--corpus_key", default="domain_corpus")
    parser.add_argument("--shard_size", type=int, default=100000)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    in_path = cfg["paths"][args.corpus_key]
    out_dir = os.path.join(os.path.dirname(in_path), "shards")
    shard_jsonl(in_path, out_dir, shard_size=args.shard_size)

if __name__ == "__main__":
    main()
