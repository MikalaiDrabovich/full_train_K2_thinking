#!/usr/bin/env python
import argparse
import json
import os
import yaml
import datasets
import torch
from transformers import (
    AutoTokenizer,
    AutoModelForCausalLM,
    TrainingArguments,
    Trainer,
)

def load_distill_data(path):
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

    teacher_path = cfg["teacher_model_path"]
    student_name = cfg["student_model_name"]
    output_dir = cfg["output_dir"]
    train_cfg = cfg["training"]

    dataset = load_distill_data(cfg["distill_data"])

    teacher_tok = AutoTokenizer.from_pretrained(teacher_path, use_fast=True)
    stud_tok = AutoTokenizer.from_pretrained(student_name, use_fast=True)
    if stud_tok.pad_token is None:
        stud_tok.pad_token = stud_tok.eos_token

    def tokenize_fn(examples):
        prompts = [e["prompt"] for e in examples]
        responses = [e["response"] for e in examples]
        texts = [
            f"<|user|>\n{p}\n<|assistant|>\n{r}"
            for p, r in zip(prompts, responses)
        ]
        return stud_tok(
            texts,
            truncation=True,
            max_length=train_cfg["max_seq_length"],
            padding="max_length",
        )

    tokenized = dataset.map(
        tokenize_fn,
        batched=True,
        remove_columns=dataset.column_names,
        desc="Tokenizing distill data",
    )

    student = AutoModelForCausalLM.from_pretrained(
        student_name,
        torch_dtype="auto",
    )

    teacher = AutoModelForCausalLM.from_pretrained(
        teacher_path,
        torch_dtype="auto",
    ).eval()
    teacher.requires_grad_(False)

    class DistillTrainer(Trainer):
        def compute_loss(self, model, inputs, return_outputs=False):
            labels = inputs["input_ids"].clone()
            outputs_student = model(**inputs)
            student_logits = outputs_student.logits

            with torch.no_grad():
                outputs_teacher = teacher(**inputs)
                teacher_logits = outputs_teacher.logits

            loss_fct = torch.nn.CrossEntropyLoss(ignore_index=-100)
            vocab_size = student_logits.size(-1)

            student_logits_shifted = student_logits[:, :-1, :].contiguous()
            teacher_logits_shifted = teacher_logits[:, :-1, :].contiguous()
            labels_shifted = labels[:, 1:].contiguous()

            ce_loss = loss_fct(
                student_logits_shifted.view(-1, vocab_size),
                labels_shifted.view(-1),
            )

            kl_loss = torch.nn.functional.kl_div(
                torch.nn.functional.log_softmax(student_logits_shifted, dim=-1),
                torch.nn.functional.softmax(teacher_logits_shifted, dim=-1),
                reduction="batchmean",
            )

            loss = ce_loss + train_cfg["kl_weight"] * kl_loss
            return (loss, outputs_student) if return_outputs else loss

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

    trainer = DistillTrainer(
        model=student,
        args=train_args,
        train_dataset=tokenized,
        tokenizer=stud_tok,
    )

    trainer.train()
    trainer.save_model(output_dir)
    stud_tok.save_pretrained(output_dir)

if __name__ == "__main__":
    main()
