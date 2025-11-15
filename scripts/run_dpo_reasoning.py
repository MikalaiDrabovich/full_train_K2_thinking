#!/usr/bin/env python
import argparse
import json
import os
import yaml
import datasets
from transformers import AutoTokenizer, AutoModelForCausalLM, TrainingArguments
from trl import DPOTrainer

def load_dpo_data(path):
    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    return datasets.Dataset.from_generator(gen)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    policy_path = cfg["policy_model_path"]
    ref_path = cfg["reference_model_path"]
    output_dir = cfg["output_dir"]
    train_cfg = cfg["training"]

    dataset = load_dpo_data(cfg["dpo_data"])

    tokenizer = AutoTokenizer.from_pretrained(policy_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def format_pair(ex):
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

if __name__ == "__main__":
    main()
