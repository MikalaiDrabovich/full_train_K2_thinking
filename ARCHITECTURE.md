# System Architecture Overview – K2 Lab

This document describes the high-level architecture of the repo and how
the pieces fit together to build smarter models on top of Kimi K2.

## 1. Components

### 1.1 Models

- **Teacher models (K2 family)**
  - `Kimi-K2-Base`: base MoE model used for continual pretraining.
  - `Kimi-K2-Instruct`: instruction-tuned version; used as a reference.
  - `Kimi-K2-Thinking` or K2-Think++: reasoning / tool-use variant.

- **Student models**
  - Smaller dense models (e.g., Qwen2.5-7B, 32B) used as distillation
    targets to obtain efficient models that approximate K2 behavior.

### 1.2 Data pipelines

- **Domain corpus**
  - General web + long-form + domain-specific text.
  - Prepared as JSONL (`{"text": "...", "source": "..."}`) via
    `scripts/prepare_corpus.py --mode domain`.

- **Reasoning corpus**
  - Math / logic / coding / QA data, including chain-of-thought style tasks.
  - Prepared via `scripts/prepare_corpus.py --mode reasoning`.

- **SFT corpora**
  - Domain SFT: JSONL with `{"instruction", "input", "output"}`.
  - Reasoning SFT: JSONL with `{"prompt", "reasoning", "final"}`.
  - DPO pairs: JSONL with `{"prompt", "chosen", "rejected"}`.
  - Distillation corpus: JSONL with `{"prompt", "response"}` generated
    by the teacher.

### 1.3 Training & fine-tuning stages

1. **Continual pretraining (K2++)**
   - Start from K2-Base.
   - Train on domain corpus to bias the model toward domain-specific
     distributions while preserving general capabilities.

2. **Domain SFT (K2-Domain-Instruct)**
   - Supervised fine-tuning on instruction-style examples in the domain.
   - Turns the pretrained model into a domain-assistant.

3. **Reasoning SFT (K2-Reason-SFT)**
   - Supervised fine-tuning with explicit chain-of-thought and tool-use
     style traces.
   - Teaches the model to think step-by-step and to follow reasoning
     templates.

4. **DPO / preference optimization (K2-Think++)**
   - Uses pairwise (chosen, rejected) reasoning traces.
   - Optimizes the policy to favor more correct, efficient, and preferred
     reasoning patterns.

5. **Distillation (K2-Student)**
   - Uses K2-Think++ as a teacher.
   - Trains a smaller dense student model on teacher outputs (and optionally
     on teacher logits) to approximate K2 behavior with lower latency and
     memory.

### 1.4 Serving & agent layer

- **Serving**
  - Teacher or student models can be served via vLLM (or any other engine).
  - `scripts/launch_teacher_vllm.py` shows a minimal launcher.

- **Agent demo**
  - `scripts/agentic_demo.py` implements a small REPL:
    - Prompts the user.
    - Lets the model call a toy `python` tool.
    - Prints intermediate tool results and final answers.
  - Intended as a minimal pattern for wiring tools, not production code.

## 2. Data & control flow

```text
+-------------------+        +-------------------------+
|   Raw corpora     |        |  Benchmarks / Prompts   |
| (web, books, ...) |        | (math, code, domain)    |
+---------+---------+        +-----------+-------------+
          |                               |
          v                               v
  scripts/prepare_corpus.py       Teacher model (K2-Base / K2-Thinking)
          |                               |
          v                               v
  domain.jsonl, reasoning.jsonl   distill_corpus.jsonl, dpo_pairs.jsonl
          |                               |
          +-------------------+-----------+
                              v
                 +-----------------------------+
                 |  Training stages (scripts/) |
                 |  - train_continual_pretrain |
                 |  - train_sft_domain         |
                 |  - train_sft_reasoning      |
                 |  - run_dpo_reasoning        |
                 |  - distill_to_student       |
                 +---------------+-------------+
                                 |
                                 v
           +--------------------------------------------+
           |  Models/ Checkpoints                      |
           |  - k2_domain_pretrain                     |
           |  - k2_domain_instruct                     |
           |  - k2_reason_sft / k2_think_dpo           |
           |  - k2_student_7b (and/or 32b)             |
           +--------------------+----------------------+
                                 |
                    +------------+-------------+
                    |                          |
                    v                          v
          launch_teacher_vllm.py      agentic_demo.py / your services
```

## 3. Configuration & extensibility

- **configs/env.yaml**
  - Shared config for model repo IDs, data paths, and vLLM endpoint.
- **configs/*.yaml**
  - Stage-specific training configs (batch size, LR, steps, sequence length).
- **scripts/**
  - Python scripts are intentionally small and modular:
    - Swap out dataset loading for your own.
    - Replace Trainer with Accelerate/DeepSpeed integration if needed.
    - Add logging, metrics, and checkpointing strategies.

## 4. Safety & practical notes

- All training scripts are generic; you must:
  - Respect the licenses of K2 and any datasets you use.
  - Ensure data filtering and PII handling meet your requirements.
  - Harden any tools (e.g., Python execution) before exposing them to users.
- The repo is designed as a starting point, not an end-to-end production
  system. Expect to iterate on:
  - Data quality / sampling.
  - Evaluation harness.
  - Scaling strategies for your particular hardware.
