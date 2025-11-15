#!/usr/bin/env python
import argparse
import json
import os
import yaml
import datasets
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    DataCollatorForLanguageModeling,
    TrainingArguments,
    Trainer,
)

def load_reason_sft(path):
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

def format_reason_example(ex):
    reasoning = ex.get("reasoning", "")
    final = ex.get("final", ex.get("response", ""))
    return PROMPT.format(
        prompt=ex["prompt"],
        reasoning=reasoning,
        final=final,
    )

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    base_model_path = cfg["base_model_path"]
    output_dir = cfg["output_dir"]
    train_cfg = cfg["training"]

    dataset = load_reason_sft(cfg["sft_data"])

    tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def tokenize_fn(examples):
        texts = [format_reason_example(e) for e in examples]
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
        desc="Tokenizing reasoning SFT",
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

if __name__ == "__main__":
    main()
