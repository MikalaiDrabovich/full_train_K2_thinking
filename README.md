# Building Smarter Models on Top of Kimi K2

This repo is a scaffold to experiment with Kimi K2 open weights and build
domain-specialist, reasoning-boosted, and distilled student models.

## Steps overview

Each root-level shell script is named `step_N_<essence>.sh` and is meant
to be run from the repo root in numerical order (you can of course skip
steps you don't need):

1. `step_1_setup_env.sh` – create Python env & install dependencies
2. `step_2_download_k2.sh` – download K2 weights from Hugging Face
3. `step_3_prepare_corpora.sh` – prepare domain & reasoning corpora
4. `step_4_continual_pretrain.sh` – continual pretraining (K2++)
5. `step_5_domain_sft.sh` – domain instruction SFT
6. `step_6_reasoning_sft.sh` – reasoning / CoT SFT
7. `step_7_dpo_reasoning.sh` – DPO preference optimization (K2-Think++)
8. `step_8_distill_student.sh` – distillation into a smaller student model
9. `step_9_evaluate.sh` – quick sanity-check evaluation
10. `step_10_agentic_demo.sh` – very small local agent demo with a Python tool

## Usage (high level)

1. Edit `configs/env.yaml` to point at:
- Your Hugging Face token (if required)
- Desired K2 model repo IDs
- Student model names
- Data and output directories

2. Adjust the YAML configs in `configs/` (batch sizes, steps, LR, etc.)
to match your hardware.

3. Run the shell scripts in order, e.g.:

```bash
bash step_1_setup_env.sh
bash step_2_download_k2.sh
bash step_3_prepare_corpora.sh
# edit data / configs as needed
bash step_4_continual_pretrain.sh
...
```

All scripts are intentionally simple and meant as starting points.
You'll almost certainly want to:
- Replace the placeholder dataset loading with your own sources
- Integrate your own logging, monitoring, and cluster launcher
- Tighten security for any tools (e.g., Python REPL) in the agent demo