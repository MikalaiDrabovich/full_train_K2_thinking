#!/usr/bin/env python
"""Prepare domain and reasoning corpora for K2 training.

This is *intentionally opinionated but minimal*:

- For `--mode domain`, it currently pulls from Wikitext-103 as a stand‑in
  for a larger web / book mixture. In a real system, you'd:
    * add FineWeb / books / your internal docs
    * add quality filtering and PII stripping

- For `--mode reasoning`, it uses GSM8K as a placeholder. In reality, you
  will likely add:
    * math benchmarks (MATH, AIME-style sets)
    * code problems
    * your own domain reasoning tasks

The output format is JSONL to keep things simple and tool‑agnostic.
"""

import argparse
import json
import os
from typing import Dict, List

import yaml
from datasets import load_dataset
from tqdm import tqdm


def write_jsonl(path: str, rows: List[Dict]) -> None:
    """Write a list of dictionaries as JSONL.

    We deliberately avoid streaming writes here because the corpora sizes
    in this scaffold are moderate. For very large corpora, you'd refill
    this function with streaming and chunked writes.
    """
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


# ---- Domain corpus builders -------------------------------------------------


def build_domain_corpus(cfg: Dict, max_examples: int | None = None) -> None:
    """Build a *placeholder* domain corpus.

    Current implementation:
    - uses `wikitext-103-v1` as a light, public example;
    - drops empty lines;
    - writes `{"text": ..., "source": "wikitext-103"}` rows.

    Replace this function with:
    - FineWeb / FineWeb-Edu samples
    - Public-domain books (Gutenberg, Harvard release, etc.)
    - Your own curated domain corpora
    """
    rows: List[Dict] = []

    ds = load_dataset("wikitext", "wikitext-103-v1", split="train")
    for i, ex in enumerate(tqdm(ds, desc="domain-wiki")):
        if max_examples is not None and i >= max_examples:
            break
        text = ex["text"].strip()
        if not text:
            continue
        rows.append({"text": text, "source": "wikitext-103"})

    out_path = cfg["paths"]["domain_corpus"]
    write_jsonl(out_path, rows)
    print(f"[prepare_corpus] Wrote {len(rows)} domain examples to {out_path}")


# ---- Reasoning corpus builders ----------------------------------------------


def build_reasoning_corpus(cfg: Dict, max_examples: int | None = None) -> None:
    """Build a *placeholder* reasoning corpus.

    Current implementation:
    - uses GSM8K as a single source of math word problems;
    - stores data as:
      `{"question": q, "answer": a, "source": "gsm8k"}`.

    In a realistic system you would:
    - concatenate from multiple math / code / logic datasets;
    - tag difficulty / source for later sampling;
    - maybe expand answers into chain-of-thought using a teacher model.
    """
    rows: List[Dict] = []

    gsm = load_dataset("gsm8k", "main", split="train")
    for i, ex in enumerate(tqdm(gsm, desc="gsm8k")):
        if max_examples is not None and i >= max_examples:
            break
        q = ex["question"].strip()
        a = ex["answer"].strip()
        if not q or not a:
            continue
        rows.append({"question": q, "answer": a, "source": "gsm8k"})

    out_path = cfg["paths"]["reasoning_corpus"]
    write_jsonl(out_path, rows)
    print(f"[prepare_corpus] Wrote {len(rows)} reasoning examples to {out_path}")


# ---- CLI entrypoint ---------------------------------------------------------


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml (for data paths).",
    )
    parser.add_argument(
        "--mode",
        choices=["domain", "reasoning"],
        required=True,
        help="Which corpus to build.",
    )
    parser.add_argument(
        "--max-examples",
        type=int,
        default=None,
        help="Optional cap on the number of examples (for quick tests).",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)

    if args.mode == "domain":
        build_domain_corpus(cfg, max_examples=args.max_examples)
    else:
        build_reasoning_corpus(cfg, max_examples=args.max_examples)


if __name__ == "__main__":
    main()
