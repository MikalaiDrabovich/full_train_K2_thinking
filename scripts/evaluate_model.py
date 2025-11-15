#!/usr/bin/env python
import argparse
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch

PROMPTS = [
    "Explain block floating point with an example.",
    "Solve: A train travels 120 km in 2 hours. What is its average speed in km/h?",
    "Write a Python function that merges two sorted lists.",
]

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        _ = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    for p in PROMPTS:
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
