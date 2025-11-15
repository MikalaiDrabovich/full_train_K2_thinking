#!/usr/bin/env bash
#
# aws_step_1_run_full_pipeline_tmux.sh
#
# Launch the full K2 Lab pipeline inside a tmux session on an AWS instance.
#
# Intent:
#   - Provide a "one command" entry point from an 8×H100 node.
#   - Run all heavy steps in a persistent tmux session so they survive
#     SSH disconnects.
#
# Assumptions:
#   - You have already unzipped / cloned the full_train_K2_thinking repo on the instance.
#   - You have activated the desired Python/conda environment.
#   - CUDA / drivers are correctly installed (DLAMI recommended).
#
# Usage:
#   bash aws_step_1_run_full_pipeline_tmux.sh
#
#   # then, to monitor:
#   tmux attach -t k2lab
#
#   # to detach without stopping the run:
#   Ctrl-b d
#

set -euo pipefail

SESSION_NAME="k2lab"
REPO_DIR="${K2LAB_REPO_DIR:-$HOME/full_train_K2_thinking}"

if [ ! -d "$REPO_DIR" ]; then
  echo "[aws_tmux] ERROR: repo dir '$REPO_DIR' does not exist."
  echo "           Set K2LAB_REPO_DIR or adjust this script."
  exit 1
fi

echo "[aws_tmux] Using repo dir: $REPO_DIR"
echo "[aws_tmux] Creating tmux session: $SESSION_NAME"

# Build the command string to run inside tmux.
PIPELINE_CMD=$(cat <<'EOF'
set -euo pipefail
cd "$REPO_DIR"

echo "[pipeline] Step 1: setup Python environment"
bash step_1_setup_env.sh

echo "[pipeline] Step 2: download K2 models (this may take a while)"
bash step_2_download_k2.sh

echo "[pipeline] Step 3: prepare corpora"
bash step_3_prepare_corpora.sh

echo "[pipeline] Step 4: continual pretraining (K2++)"
bash step_4_continual_pretrain.sh

echo "[pipeline] Step 5: domain SFT"
bash step_5_domain_sft.sh

echo "[pipeline] Step 6: reasoning SFT"
bash step_6_reasoning_sft.sh

echo "[pipeline] Step 7: DPO reasoning"
bash step_7_dpo_reasoning.sh

echo "[pipeline] Step 8: distill student"
bash step_8_distill_student.sh

echo "[pipeline] Step 9: evaluate student"
bash step_9_evaluate.sh

echo "[pipeline] All steps completed."
exec bash
EOF
)

# Export REPO_DIR into the tmux environment so the inner script sees it.
tmux new-session -d -s "$SESSION_NAME" "export REPO_DIR='$REPO_DIR'; $PIPELINE_CMD"

echo "[aws_tmux] tmux session '$SESSION_NAME' started."
echo "           Attach with: tmux attach -t $SESSION_NAME"
