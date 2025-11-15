#!/usr/bin/env python
"""Very small agentic REPL demo with a Python tool.

This is NOT production-safe. It is meant only as a pattern to show:

- how to frame a system prompt that describes tools;
- how to let the model emit tool calls in a simple XML-ish format;
- how to execute a `python` tool and feed results back to the model.

**IMPORTANT**
The `python` tool uses `exec` and is unsafe by default. Do not expose this
beyond a controlled environment without sandboxing.
"""

import argparse
from typing import Dict

import yaml
from transformers import AutoTokenizer, AutoModelForCausalLM
import torch


def exec_python(code: str) -> str:
    """Execute Python code in a tiny local namespace.

    This is intentionally minimal and not sandboxed. Replace this with
    a real sandbox or restricted executor if you plan to use it seriously.
    """
    local_env: Dict = {}
    try:
        exec(code, {}, local_env)
        return repr(local_env)
    except Exception as e:  # noqa: BLE001
        return f"Error: {e}"


SYSTEM_PROMPT = """You are a planning agent.
You have access to tools:
- python(code: str) -> result

When needed, call tools using the syntax:

<tool name="python">
print(1+2)
</tool>

Then respond to the user with your final answer after you see the tool output.
Keep your reasoning concise and focus on correct results.
"""


def generate_response(model, tokenizer, prompt: str, max_new_tokens: int = 512) -> str:
    """Helper to call `generate` with sane defaults."""
    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(out[0], skip_special_tokens=False)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        required=True,
        help="Path to env.yaml (not heavily used here).",
    )
    parser.add_argument(
        "--model_path",
        default="./models/k2_student_7b",
        help="Path to the student or teacher model to drive the agent.",
    )
    args = parser.parse_args()

    with open(args.config, "r", encoding="utf-8") as f:
        _cfg = yaml.safe_load(f)

    tok = AutoTokenizer.from_pretrained(args.model_path, use_fast=True)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    model = AutoModelForCausalLM.from_pretrained(
        args.model_path,
        torch_dtype="auto",
        device_map="auto",
    )

    print("Agent REPL ready. Ctrl-C to exit.")
    while True:
        user = input("\nUser: ")
        if not user.strip():
            continue

        # First turn: ask model for a response (possibly with a tool call)
        convo = SYSTEM_PROMPT + f"\nUser: {user}\nAssistant:"
        assistant = generate_response(model, tok, convo)

        # Simple, single-tool parsing loop.
        if "<tool" in assistant and "</tool>" in assistant:
            pre, rest = assistant.split("<tool", 1)
            print("Assistant (before tool):", pre.strip())
            tag_body, after = rest.split("</tool>", 1)
            # Very naive parsing; looks for name="python"
            if 'name="python"' in tag_body:
                code = tag_body.split(">", 1)[1]
                tool_result = exec_python(code)
                print("[tool:python result]", tool_result)

                # Feed tool result back to the model and ask it to conclude.
                followup_prompt = (
                    convo
                    + assistant
                    + f"\n[tool:python_result]\n{tool_result}\n"
                    + "\nAssistant (final answer):"
                )
                final = generate_response(model, tok, followup_prompt, max_new_tokens=256)
                final_resp = final.split("Assistant (final answer):", 1)[-1]
                print("Assistant:", final_resp.strip())
            else:
                # Some other tool; in this scaffold, just print the raw output.
                print("Assistant:", assistant.strip())
        else:
            # No tool usage, just print the answer.
            print("Assistant:", assistant.strip())


if __name__ == "__main__":
    main()
