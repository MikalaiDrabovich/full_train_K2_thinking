#!/usr/bin/env python
"""Run DPO (Direct Preference Optimization) for reasoning.

Expects a JSONL file where each line looks like:

    {
      "prompt": "...",
      "chosen": "...",    # preferred reasoning trace
      "rejected": "..."   # less preferred trace
    }

This script:
- loads a policy model (K2-Reason-SFT);
- loads a reference model (typically K2-Domain-Instruct);
- optimizes the policy to upweight the chosen traces vs rejected ones.

The goal is to push the model toward better reasoning style and accuracy.
"""

import argparse
import json
from typing import Dict

import datasets
import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
# TRL implements DPOTrainer for us
from trl import DPOTrainer


def load_dpo_data(path: str) -> datasets.Dataset:
    """Load DPO preference data from a JSONL file."""
    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    return datasets.Dataset.from_generator(gen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/dpo_reasoning.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    policy_path = cfg["policy_model_path"]
    ref_path = cfg["reference_model_path"]
    output_dir = cfg["output_dir"]
    train_cfg = cfg["training"]
    dpo_path = cfg["dpo_data"]

    dataset = load_dpo_data(dpo_path)

    tokenizer = AutoTokenizer.from_pretrained(policy_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_pair(ex: Dict) -> Dict:
        """Normalize a raw JSON object into the fields DPOTrainer expects."""
        return {
            "prompt": ex["prompt"],
            "chosen": ex["chosen"],
            "rejected": ex["rejected"],
        }

    dataset = dataset.map(format_pair)

    policy_model = AutoModelForCausalLM.from_pretrained(
        policy_path,
        torch_dtype="auto",
    )

    ref_model = AutoModelForCausalLM.from_pretrained(
        ref_path,
        torch_dtype="auto",
    )

    train_args = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        num_train_epochs=train_cfg["num_train_epochs"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        bf16=train_cfg.get("bf16", True),
        report_to=["none"],
    )

    dpo_trainer = DPOTrainer(
        model=policy_model,
        ref_model=ref_model,
        args=train_args,
        beta=train_cfg["beta"],
        train_dataset=dataset,
        tokenizer=tokenizer,
        max_length=train_cfg["max_seq_length"],
    )

    dpo_trainer.train()
    dpo_trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[run_dpo_reasoning] Saved DPO-tuned model to {output_dir}")


if __name__ == "__main__":
    main()
