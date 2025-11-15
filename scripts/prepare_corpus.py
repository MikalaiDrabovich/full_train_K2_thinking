#!/usr/bin/env python
import argparse
import json
import os
import yaml
from datasets import load_dataset
from tqdm import tqdm

def write_jsonl(path, rows):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

def build_domain_corpus(cfg):
    rows = []
    # Placeholder example: Wikitext-103; replace with your own sources.
    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    for ex in tqdm(ds, desc="domain-wiki"):
        text = ex["text"].strip()
        if not text:
            continue
        rows.append({"text": text, "source": "wikitext-103"})
    out_path = cfg["paths"]["domain_corpus"]
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} domain examples to {out_path}")

def build_reasoning_corpus(cfg):
    rows = []
    # Placeholder example: GSM8K; extend with math/code/logic datasets.
    gsm = load_dataset("gsm8k", "main", split="train")
    for ex in tqdm(gsm, desc="gsm8k"):
        q = ex["question"].strip()
        a = ex["answer"].strip()
        rows.append({"question": q, "answer": a, "source": "gsm8k"})
    out_path = cfg["paths"]["reasoning_corpus"]
    write_jsonl(out_path, rows)
    print(f"Wrote {len(rows)} reasoning examples to {out_path}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--mode", choices=["domain", "reasoning"], required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "domain":
        build_domain_corpus(cfg)
    else:
        build_reasoning_corpus(cfg)

if __name__ == "__main__":
    main()
