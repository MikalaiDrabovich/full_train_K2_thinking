    #!/usr/bin/env python
    """Supervised fine-tuning (SFT) for domain-specific instructions.

    This script expects a JSONL file with entries:

        {"instruction": "...", "input": "...", "output": "..."}

    and formats them into a simple chat-style prompt:

        <|user|>
        {instruction}

{input}
        <|assistant|>
        {output}

    This is deliberately simple and should be adapted to match the exact
    chat/instruction format used by your K2 variant.
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


    def load_sft_dataset(path: str) -> datasets.Dataset:
        """Load a JSONL SFT dataset into a Hugging Face `Dataset`."""
        def gen():
            with open(path, "r", encoding="utf-8") as f:
                for line in f:
                    ex = json.loads(line)
                    yield ex
        return datasets.Dataset.from_generator(gen)


    PROMPT_TEMPLATE = """<|user|>
    {instruction}{input_block}
    <|assistant|>
    {output}"""


    def format_example(ex: Dict) -> str:
        """Format a single SFT example into a textual prompt."""
        input_block = ""
        if ex.get("input"):
            input_block = "\n\n" + ex["input"]
        return PROMPT_TEMPLATE.format(
            instruction=ex["instruction"],
            input_block=input_block,
            output=ex["output"],
        )


    def main() -> None:
        parser = argparse.ArgumentParser()
        parser.add_argument(
            "--config",
            required=True,
            help="Path to configs/sft_domain.yaml",
        )
        args = parser.parse_args()

        with open(args.config, "r", encoding="utf-8") as f:
            cfg: Dict = yaml.safe_load(f)

        base_model_path = cfg["base_model_path"]
        output_dir = cfg["output_dir"]
        train_cfg = cfg["training"]
        sft_path = cfg["sft_data"]

        os.makedirs(output_dir, exist_ok=True)

        dataset = load_sft_dataset(sft_path)

        tokenizer = AutoTokenizer.from_pretrained(base_model_path, use_fast=True)
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token

        def tokenize_fn(batch: Dict) -> Dict:
            """Tokenize a batch of formatted SFT prompts."""
            texts = [format_example(e) for e in batch]
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
            desc="Tokenizing domain SFT data",
        )

        model = AutoModelForCausalLM.from_pretrained(
            base_model_path,
            torch_dtype="auto",
            device_map=None,
        )

        data_collator = DataCollatorForLanguageModeling(
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
            data_collator=data_collator,
        )

        trainer.train()
        trainer.save_model(output_dir)
        tokenizer.save_pretrained(output_dir)
        print(f"[train_sft_domain] Saved model to {output_dir}")


    if __name__ == "__main__":
        main()
