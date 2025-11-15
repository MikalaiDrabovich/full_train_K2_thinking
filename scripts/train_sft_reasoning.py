#!/usr/bin/env python
"""Reasoning / chain-of-thought Supervised Fine-Tuning.

Expects a JSONL file where each line is an object of the form:

    {
      "prompt": "...",
      "reasoning": "...",   # optional long chain-of-thought
      "final": "..."        # final short answer
    }

We wrap this into a simple template where reasoning is explicitly marked:

    <|user|>
    {prompt}
    <|assistant|>
    <think>
    {reasoning}
    </think>
    {final}
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
    TrainingArguments,
    Trainer,
)


def load_reason_sft(path: str) -> datasets.Dataset:
    """Load the reasoning SFT dataset from a JSONL file."""
    def gen():
        with open(path, "r", encoding="utf-8") as f:
            for line in f:
                yield json.loads(line)
    return datasets.Dataset.from_generator(gen)


PROMPT = """<|user|>
{prompt}
<|assistant|>
<think>
{reasoning}
</think>
{final}"""


def format_reason_example(ex: Dict) -> str:
    """Format a single reasoning example."""
    reasoning = ex.get("reasoning", "")
    final = ex.get("final", ex.get("response", ""))
    return PROMPT.format(
        prompt=ex["prompt"],
        reasoning=reasoning,
        final=final,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to configs/sft_reasoning.yaml",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        cfg: Dict = yaml.safe_load(f)

    base_model_path = cfg["base_model_path"]
    output_dir = cfg["output_dir"]
    train_cfg = cfg["training"]
    sft_path = cfg["sft_data"]

    dataset = load_reason_sft(sft_path)

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(batch: Dict) -> Dict:
        """Tokenize a batch of reasoning examples."""
        texts = [format_reason_example(e) for e in batch]
        return tokenizer(
            texts,
            truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing reasoning SFT data",
    )

    model = AutoModelForCausalLM.from_pretrained(
        base_model_path,
        torch_dtype="auto",
    )

    collator = DataCollatorForLanguageModeling(
        tokenizer=tokenizer,
        mlm=False,
    )

    args_train = TrainingArguments(
        output_dir=output_dir,
        num_train_epochs=train_cfg["num_train_epochs"],
        per_device_train_batch_size=train_cfg["per_device_train_batch_size"],
        gradient_accumulation_steps=train_cfg["gradient_accumulation_steps"],
        learning_rate=train_cfg["learning_rate"],
        warmup_steps=train_cfg["warmup_steps"],
        weight_decay=train_cfg["weight_decay"],
        logging_steps=train_cfg["logging_steps"],
        save_steps=train_cfg["save_steps"],
        bf16=train_cfg.get("bf16", True),
        report_to=["none"],
    )

    trainer = Trainer(
        model=model,
        args=args_train,
        train_dataset=tokenized,
        data_collator=collator,
    )

    trainer.train()
    trainer.save_model(output_dir)
    tokenizer.save_pretrained(output_dir)
    print(f"[train_sft_reasoning] Saved model to {output_dir}")


if __name__ == "__main__":
    main()
