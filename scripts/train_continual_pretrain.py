#!/usr/bin/env python
"""Continual pretraining of K2-Base on a domain corpus.

This script performs *full-parameter* continued pretraining using
Hugging Face `Trainer`. It is intentionally straightforward:

- It consumes a JSONL corpus with a `text` field.
- It tokenizes with the K2 tokenizer.
- It trains with a standard next-token prediction objective.

For serious scale you will likely:
- switch to `accelerate` + DeepSpeed or FSDP;
- implement streaming datasets instead of loading everything in memory;
- track validation perplexity on held-out splits.
"""

import argparse
import json
import os
from typing import Dict

import datasets
import yaml
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)


def load_jsonl_as_hf_dataset(path: str, text_key: str = "text") -> datasets.Dataset:
    """Load a JSONL file into a Hugging Face `Dataset`.

    Each line is expected to be a JSON object containing `text_key`.
    """
    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                obj = json.loads(line)
                if text_key not in obj:
                    continue
                yield {text_key: obj[text_key]}
    return datasets.Dataset.from_generator(gen)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/continual_pretrain.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    base_model_path = cfg["base_model_path"]
    output_dir = cfg["output_dir"]
    train_path = cfg["train_corpus"]
    train_cfg = cfg["training"]

    os.makedirs(output_dir, exist_ok=True)

    # Load tokenizer and ensure we have a pad token
    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Load dataset
    dataset = load_jsonl_as_hf_dataset(train_path, text_key="text")

    # Optional: shuffle for better mixing (for large corpora, use streaming)
    dataset = dataset.shuffle(seed=42)

    def tokenize_fn(examples: Dict) -> Dict:
        """Tokenize a batch of raw text examples."""
        return tokenizer(
            examples["text"],
            truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=["text"],
        desc="Tokenizing domain corpus",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
        device_map=None,  # let torchrun / deepspeed handle placement
    )

    data_collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args_train = TrainingArguments(
        output_dir=output_dir,
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        max_steps=train_cfg["max_steps"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        bf16=train_cfg.get("bf16", True),
        gradient_checkpointing=train_cfg.get("gradient_checkpointing", True),
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
        data_collator=data_collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_continual_pretrain] Saved model to {output_dir}")


if __name__ == "__main__":
    main()
