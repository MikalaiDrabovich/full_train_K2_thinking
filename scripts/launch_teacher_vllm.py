#!/usr/bin/env python
import argparse
import yaml
import subprocess

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--model_path", default="./models/k2_think_dpo")
    args = parser.parse_args()

    with open(args.config, "r") as f:
        cfg = yaml.safe_load(f)

    host = cfg["vllm"]["host"]
    port = cfg["vllm"]["port"]

    cmd = [
        "python", "-m", "vllm.entrypoints.openai.api_server",
        "--model", args.model_path,
        "--host", host,
        "--port", str(port),
    ]
    print("Launching vLLM server:", " ".join(cmd))
    subprocess.run(cmd)

if __name__ == "__main__":
    main()
