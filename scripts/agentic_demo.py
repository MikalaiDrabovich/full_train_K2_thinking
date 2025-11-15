#!/usr/bin/env python
import argparse
import yaml
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM

def exec_python(code: str) -> str:
    # WARNING: This is not sandboxed. Do NOT expose directly to untrusted users.
    local_env = {}
    try:
        exec(code, {}, local_env)
        return repr(local_env)
    except Exception as e:
        return f"Error: {e}"

SYSTEM_PROMPT = """You are a planning agent.
You have access to tools:
- python(code: str) -> result

When needed, call tools using the syntax:
<tool name="python">
print(1+2)
</tool>

Then respond to the user with your final answer after the tool output.
"""

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="./models/k2_student_7b")
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

    print("Agent ready. Ctrl-C to exit.")
    while True:
        user = input("\nUser: ")
        if not user.strip():
            continue

        prompt = SYSTEM_PROMPT + f"\nUser: {user}\nAssistant:"
        inputs = tok(prompt, return_tensors="pt").to(model.device)

        with torch.no_grad():
            out = model.generate(
                **inputs,
                max_new_tokens=512,
                do_sample=True,
                temperature=0.7,
                top_p=0.9,
            )

        text = tok.decode(out[0], skip_special_tokens=False)
        assistant = text.split("Assistant:", 1)[-1]

        # naive one-shot display (no real tool loop parsing)
        print("Assistant:", assistant.strip())

if __name__ == "__main__":
    main()
