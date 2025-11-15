#!/usr/bin/env python
"""Quick smoke-test evaluation for a model.

This script is NOT a benchmark harness. It is designed to:

- load a local model and tokenizer;
- run it on a few hard-coded prompts (or an optional JSONL file);
- print the outputs to stdout.

For real evaluation, you should:
- integrate MMLU / GSM8K / HumanEval etc.;
- compute accuracy / pass@k metrics;
- log results to a dashboard.
"""

import argparse
from typing import Dict, List

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


DEFAULT_PROMPTS: List[str] = [
    "Explain block floating point with a concrete numeric example.",
    "Solve: A train travels 120 km in 2 hours. What is its average speed in km/h?",
    "Write a Python function that merges two sorted lists into one sorted list.",
]


def load_prompts_from_jsonl(path: str) -> List[str]:
    """Optionally load prompts from a JSONL file.

    Each line is expected to have at least a `prompt` field.
    """
    import json
    prompts: List[str] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            obj = json.loads(line)
            if "prompt" in obj:
                prompts.append(obj["prompt"])
    return prompts


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml (not heavily used, but kept for consistency).",
    )
    parser.add_argument(
        "--model_path",
        required=True,
        help="Path to local model directory.",
    )
    parser.add_argument(
        "--prompts-jsonl",
        default=None,
        help="Optional JSONL file with `prompt` fields to evaluate on.",
    )
    args = parser.parse_args()

    # We don't really need env config here, but keep for symmetry.
    with open(args.config, "r", encoding="utf-8") as f:
        _cfg: Dict = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    if args.prompts_jsonl:
        prompts = load_prompts_from_jsonl(args.prompts_jsonl)
    else:
        prompts = DEFAULT_PROMPTS

    for p in prompts:
        print("=" * 80)
        print("Prompt:", p)
        inputs = tok(p, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=256,
                do_sample=False,
            )
        text = tok.decode(out[0], skip_special_tokens=True)
        print("Output:\n", text)


if __name__ == "__main__":
    main()
