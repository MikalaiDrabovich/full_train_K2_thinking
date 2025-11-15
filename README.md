# K2 Lab – Building Smarter Models on Top of Kimi K2

This repository is a **scaffold**, not a finished product. The intent is to give
you a reasonably realistic starting point for:

- experimenting with open Kimi K2 weights locally;
- doing **continual pretraining** on domain data (K2++);
- doing **instruction SFT** (Supervised Fine-Tuning) for your domain;
- doing **reasoning SFT + DPO** to get a better "thinking" variant;
- **distilling** K2 into smaller dense models that are cheap to run;
- wiring a tiny **agentic demo** with a Python tool.

It deliberately trades completeness for clarity: nearly every script is short,
well-commented, and meant to be read and modified.

---

## 0. Assumptions & design intentions

Before you run anything, it’s important to be explicit about what this repo
assumes and how it is **intended** to be used.

### Hardware / scale assumptions

- You have **at least one GPU** with enough VRAM to load a 7B model.
- Many of the shell scripts are written with `--nproc_per_node=8` as an example,
  assuming a single node with 8 GPUs (A100/H100 class). You **must** adjust these
  numbers to your actual hardware.
- Hugging Face `Trainer` is used for simplicity. For serious scale you will likely
  switch to:
  - `accelerate` + DeepSpeed / FSDP;
  - streaming datasets instead of loading everything into RAM at once.

### Data / licensing assumptions

- This repo does **not** ship any data; it only contains convenience code to
  pull some **public placeholder datasets** (Wikitext, GSM8K).
- You are responsible for:
  - choosing the actual datasets you want (FineWeb, books, internal corpora, ...);
  - checking licenses and usage rights for both K2 and any data you use;
  - adding PII filtering and other safety mechanisms where appropriate.

### Modeling / training philosophy

- The goal is **practical experimentation**, not reproducing K2’s full training.
- Each stage is split into:
  - a config file under `configs/`;
  - a Python script under `scripts/`;
  - a shell wrapper `step_N_*.sh` at the repo root.
- You should treat these stages as **lego bricks**:
  - plug in your own data sources;
  - tweak hyperparameters;
  - replace parts (e.g. SFT format) to match your chat template.

---

## 1. Repo layout

```text
full_train_K2_thinking/
  README.md
  REAMDE.txt
  ARCHITECTURE.md
  requirements.txt

  configs/
    env.yaml
    continual_pretrain.yaml
    sft_domain.yaml
    sft_reasoning.yaml
    dpo_reasoning.yaml
    distill_student.yaml

  scripts/
    download_k2.py
    prepare_corpus.py
    shard_corpus.py
    train_continual_pretrain.py
    train_sft_domain.py
    train_sft_reasoning.py
    run_dpo_reasoning.py
    distill_to_student.py
    launch_teacher_vllm.py
    evaluate_model.py
    agentic_demo.py

  step_1_setup_env.sh
  step_2_download_k2.sh
  step_3_prepare_corpora.sh
  step_4_continual_pretrain.sh
  step_5_domain_sft.sh
  step_6_reasoning_sft.sh
  step_7_dpo_reasoning.sh
  step_8_distill_student.sh
  step_9_evaluate.sh
  step_10_agentic_demo.sh
```

- `README.md` – this document.
- `ARCHITECTURE.md` – system architecture overview and data/compute flow.
- `configs/` – training and environment configs (edit these first).
- `scripts/` – Python entrypoints; each script has docstrings and comments
  explaining what it does and what you’re expected to customize.
- `step_*.sh` – linear “happy path” scripts so you can run each stage with
  a single shell command.

---

## 2. Step-by-step pipeline (with intentions)

Each step has a root-level shell script named
`step_N_(brief_essence of what it does).sh`. The idea is **traceable stages**
rather than one opaque mega-script.

### Step 1 – Environment setup

**Script:** `step_1_setup_env.sh`  
**Python:** `requirements.txt`

```bash
bash step_1_setup_env.sh
```

**What it does**
- Upgrades `pip` and installs the Python packages needed by the scripts:
  - `torch`, `transformers`, `datasets`, `trl`, `vllm`, etc.

**Assumptions & intentions**
- You already created and activated a virtual environment (conda, venv, etc.).
- You may want to pin versions more tightly in `requirements.txt` once you know
  what works best on your hardware / CUDA stack.

---

### Step 2 – Download K2 teacher weights

**Script:** `step_2_download_k2.sh`  
**Python:** `scripts/download_k2.py` + `configs/env.yaml`

```bash
bash step_2_download_k2.sh
```

**What it does**
- Reads `teacher_models` from `configs/env.yaml`, e.g.:

  ```yaml
  teacher_models:
    k2_base: "moonshotai/Kimi-K2-Base"
    k2_instruct: "moonshotai/Kimi-K2-Instruct"
    k2_thinking: "moonshotai/Kimi-K2-Thinking"
  ```

- Uses `huggingface_hub.snapshot_download` to pull each model into `models/`.

**Assumptions & intentions**
- You have a valid Hugging Face token in `hf_token` if the repos are gated.
- You will likely only run this occasionally; it is not meant for tight
  inner loops.
- If you use alternative teacher models, simply update `env.yaml`.

---

### Step 3 – Prepare corpora

**Script:** `step_3_prepare_corpora.sh`  
**Python:** `scripts/prepare_corpus.py`, `scripts/shard_corpus.py`

```bash
bash step_3_prepare_corpora.sh
```

**What it does**
- For now, builds two **placeholder** corpora:
  - `domain.jsonl` from Wikitext-103 (as a stand-in for web/books).
  - `reasoning.jsonl` from GSM8K (as a stand-in for math reasoning).

- Each JSONL file has simple, explicit fields:
  - Domain corpus: `{"text": "...", "source": "wikitext-103"}`
  - Reasoning corpus: `{"question": "...", "answer": "...", "source": "gsm8k"}`

- `shard_corpus.py` can then split big corpora into multiple shards.

**Assumptions & intentions**
- These are deliberate placeholders. **You are expected to replace them** by:
  - plugging in your own datasets in `prepare_corpus.py`;
  - adding deduplication / quality / PII filters;
  - tagging domain / difficulty for smarter sampling.
- The scaffolding ensures you don’t also have to write boilerplate to handle
  JSONL, sharding, etc.

---

### Step 4 – Continual pretraining (K2++)

**Script:** `step_4_continual_pretrain.sh`  
**Python:** `scripts/train_continual_pretrain.py`  
**Config:** `configs/continual_pretrain.yaml`

```bash
bash step_4_continual_pretrain.sh
```

**What it does**
- Loads `Kimi-K2-Base` from `./models/k2_base`.
- Loads `domain.jsonl` and tokenizes it.
- Runs full-parameter continued pretraining on your domain corpus.

**Why this step exists**
- It nudges K2’s prior toward your domain distribution *without* throwing away
  its general knowledge.
- This is often a better starting point for domain SFT than the original base.

**Key assumptions**
- You will dial `max_steps`, `learning_rate`, `max_seq_length`, etc. to match
  your hardware and corpus size.
- For real runs, you will likely want:
  - validation splits and early stopping;
  - logging to WandB / TensorBoard;
  - DeepSpeed / FSDP instead of vanilla `Trainer`.

---

### Step 5 – Domain instruction SFT

**Script:** `step_5_domain_sft.sh`  
**Python:** `scripts/train_sft_domain.py`  
**Config:** `configs/sft_domain.yaml`

```bash
bash step_5_domain_sft.sh
```

**What it does**
- Takes `k2_domain_pretrain` as the base model.
- Loads a JSONL instruction dataset:

  ```json
  {"instruction": "...", "input": "...", "output": "..."}
  ```

- Formats them into a simple user/assistant prompt and fine-tunes with
  next-token prediction.

**Intentions**
- This is the stage where the model becomes a **domain assistant** that can
  follow your instructions, not just model your domain text distribution.
- The prompt template is intentionally simple; you should adapt it to your
  preferred chat format (system messages, special tokens, etc.).

**Assumptions**
- You will curate / generate your own SFT data under `data/domain_sft.jsonl`.
- You may want to mix in some general instructions so the model doesn’t
  become too narrow.

---

### Step 6 – Reasoning SFT

**Script:** `step_6_reasoning_sft.sh`  
**Python:** `scripts/train_sft_reasoning.py`  
**Config:** `configs/sft_reasoning.yaml`

```bash
bash step_6_reasoning_sft.sh
```

**What it does**
- Starts from `k2_domain_instruct`.
- Consumes reasoning examples like:

  ```json
  {
    "prompt": "...",
    "reasoning": "long chain-of-thought...",
    "final": "short final answer"
  }
  ```

- Teaches the model to emit `<think>...</think>` style reasoning traces.

**Intentions**
- This approximates what K2-Thinking does: long internal traces plus short,
  polished answers.
- You’ll likely generate this data by sampling from teacher models and
  auto-grading their outputs.

**Assumptions**
- Reasoning traces in your JSONL are already **clean and correct**; this
  script doesn’t grade them.
- You will adjust `max_seq_length` and batch sizes to handle longer traces.

---

### Step 7 – DPO reasoning (preference optimization)

**Script:** `step_7_dpo_reasoning.sh`  
**Python:** `scripts/run_dpo_reasoning.py`  
**Config:** `configs/dpo_reasoning.yaml`

```bash
bash step_7_dpo_reasoning.sh
```

**What it does**
- Loads:
  - policy model: `k2_reason_sft`
  - reference model: `k2_domain_instruct`
- Optimizes a DPO objective on pairs:

  ```json
  { "prompt": "...", "chosen": "...", "rejected": "..." }
  ```

**Intentions**
- Move beyond “imitate whatever SFT data you gave me” toward:
  - *prefer more correct / elegant / concise reasoning*;
  - *avoid common failure modes* that show up in rejected examples.

**Assumptions**
- You have a process (human or automatic) to produce chosen vs rejected
  reasoning traces.
- Hyperparameters like `beta` in `dpo_reasoning.yaml` will be tuned for
  stability and performance.

---

### Step 8 – Distillation into a smaller student

**Script:** `step_8_distill_student.sh`  
**Python:** `scripts/distill_to_student.py`  
**Config:** `configs/distill_student.yaml`

```bash
bash step_8_distill_student.sh
```

**What it does**
- Treats a heavy K2 teacher (e.g. `k2_think_dpo`) as the gold standard.
- Trains a smaller dense student (e.g. Qwen2.5‑7B) to match:
  - the teacher’s token predictions (LM loss);
  - and the teacher’s logits (KL loss).

**Intentions**
- Give you **K2-like behavior in a much smaller footprint**, suitable for
  local inference or higher QPS scenarios.
- Make it easy to swap out student architectures by editing
  `student_model_name` in `distill_student.yaml`.

**Assumptions**
- You have generated a distillation corpus via the teacher model under
  `data/distill_corpus.jsonl`.
- You will experiment with:
  - different student sizes;
  - different KL weights;
  - and possibly adding alignment / SFT data into the same training run.

---

### Step 9 – Quick evaluation

**Script:** `step_9_evaluate.sh`  
**Python:** `scripts/evaluate_model.py`

```bash
bash step_9_evaluate.sh
```

**What it does**
- Loads a given model (by default `./models/k2_student_7b`).
- Runs it on a small set of canned prompts or an optional `--prompts-jsonl`.
- Prints outputs to stdout.

**Intentions**
- This is a **smoke test**, not a benchmark.
- You can use it to sanity-check that:
  - the model loads;
  - the tokenizer matches;
  - the logit head is not completely broken.

**Assumptions**
- You will eventually build a real evaluation harness on top of this, using
  standard benchmarks and metrics.

---

### Step 10 – Agentic demo with tools

**Script:** `step_10_agentic_demo.sh`  
**Python:** `scripts/agentic_demo.py`

```bash
bash step_10_agentic_demo.sh
```

**What it does**
- Starts a command-line REPL:
  - you type user queries;
  - the model can optionally emit `<tool name="python"> ... </tool>` blocks;
  - the script executes the code in a tiny `exec` sandbox and prints results;
  - the model gets a follow-up chance to produce a final answer.

**Intentions**
- Demonstrate the basic **agent pattern**:
  - plan → tool call → observe → answer.
- Serve as a template for wiring real tools:
  - retrieval / search;
  - code execution in proper sandboxes;
  - APIs / databases, etc.

**Assumptions & warnings**
- The `python` tool is **not safe**. It uses Python `exec` with no sandbox.
  Only run it in a controlled environment.
- You will replace or harden this tool before exposing it to any users.

---

## 3. How to turn this into your own lab

1. **Fork the repo / import into GitHub.**
   - Replace placeholder datasets in `scripts/prepare_corpus.py` with your own.
   - Commit your changes so you can track experiments.

2. **Define a clear goal.**
   - “Beat K2 on SWE-bench for our codebase”
   - “Build the best finance assistant we can on 7B”
   - etc.

3. **Iterate by stage.**
   - Start with continual pretraining and domain SFT.
   - Once those are stable, add reasoning SFT + DPO.
   - Finally, distill into smaller students that actually fit your deployment.

4. **Add real evaluation early.**
   - Hook in at least one or two public benchmarks that align with your goal.
   - Use `evaluate_model.py` as a seed and build from there.

5. **Treat this as scaffolding, not gospel.**
   - Swap pieces as you learn:
     - different chat templates;
     - more sophisticated data filtering;
     - better optimizers / schedulers;
     - alternative training libraries.

This repo is meant to get you from “idea in your head” to “actual code +
configs that run” as quickly as possible, while being explicit about the
assumptions and intentions behind each step.
---
## 4. Rough AWS cost estimates (8×H100 / p5.48xlarge)

This is a **very rough** back-of-the-envelope guide to what it might cost to
run the full pipeline on a single 8×H100 AWS instance (e.g. `p5.48xlarge`).

### 4.1 On-demand hourly price (ballpark)

Public price trackers show on-demand `p5.48xlarge` pricing on the order of
**\$55/hour** in some US regions for Linux. Exact pricing depends on region,
discounts, and whether you use on-demand, Savings Plans, or Capacity Blocks.

For quick mental math, you can think:

- \$55/hour  ≈  \$1,320/day  (24 hours)
- \$55/hour  ≈  \$9,240/week (7 × 24 hours)

Always check the AWS pricing page / calculator for current numbers.

### 4.2 Example wall-clock + cost scenarios

These scenarios are *illustrative only*. Real times will depend heavily on:

- how much data you actually feed into each stage;
- how many steps/epochs you choose to run;
- I/O / data pipeline efficiency and any DeepSpeed/FSDP optimizations.

**Scenario A – Light run / smoke-tested pipeline**

Rough idea (continuous use of the node):

- Continual pretraining: ~1–2 days
- Domain SFT: ~0.5–1 day
- Reasoning SFT: ~0.5–1 day
- DPO reasoning: ~0.5 day
- Distillation + eval: ~0.5 day

Total: ~3–5 days of 8×H100 time

- 3 days  → 3 × \$1,320 ≈ **\$4k**
- 5 days  → 5 × \$1,320 ≈ **\$6.6k**

**Scenario B – Heavier training**

If you push continual pretraining and reasoning SFT harder (more steps, more
data), you could easily be in the **7–10 day** range of wall-clock time.

- 7 days   → 7 × \$1,320 ≈ **\$9.2k**
- 10 days  → 10 × \$1,320 ≈ **\$13.2k**

In practice you will probably:

- start with a shorter run to validate the pipeline end-to-end;
- instrument throughput (tokens/second, steps/hour) and scale up from there;
- use Spot / Savings Plans / Capacity Blocks if you’re doing this routinely.

### 4.3 How to tighten estimates for your setup

To get more precise numbers:

1. Decide how many **tokens** you want per stage (e.g. 30B for continual
   pretraining, 5B for domain SFT, etc.).
2. Compute **tokens per step** from the YAML configs:

   ```text
   tokens_per_step = num_gpus * per_device_batch * grad_acc * seq_len
   ```

3. Solve for `max_steps` given desired tokens, or invert the math to see how
   many tokens your current config implies.
4. Time a few hundred steps on the real instance to estimate
   **steps/hour → tokens/hour → cost**.

Treat these numbers as **order-of-magnitude planning tools**, not promises.
The goal is simply to make the costs legible enough that you can adjust
your run lengths and budgets before committing to very long jobs.
