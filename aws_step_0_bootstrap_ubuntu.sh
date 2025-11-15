#!/usr/bin/env bash
#
# aws_step_0_bootstrap_ubuntu.sh
#
# One-time bootstrap script for a FRESH Ubuntu-based AWS GPU instance.
#
# Intent:
#   - Install basic system packages (build-essential, git, tmux, etc.).
#   - Install Miniconda if not already present.
#   - Create a conda env for k2-lab (optional) and activate it manually.
#
# Assumptions:
#   - You are running as a user with sudo privileges (e.g. `ubuntu`).
#   - If you are using an AWS Deep Learning AMI (DLAMI), many of these
#     steps are optional or redundant; adjust as needed.
#
# This script is designed to be SAFE-ish to run multiple times, but you
# should still skim it before executing in case you want to change paths.

set -euo pipefail

echo "[aws_bootstrap] Updating apt package index..."
sudo apt-get update -y

echo "[aws_bootstrap] Installing common packages..."
sudo DEBIAN_FRONTEND=noninteractive apt-get install -y \
    build-essential \
    git \
    tmux \
    htop \
    wget \
    curl \
    unzip \
    pkg-config \
    python3-dev \
    python3-venv

# --- Miniconda installation (optional but handy) -------------------------#
CONDA_DIR="$HOME/miniconda3"
if [ ! -d "$CONDA_DIR" ]; then
  echo "[aws_bootstrap] Miniconda not found, installing to $CONDA_DIR..."
  wget -O /tmp/miniconda.sh \
    https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
  bash /tmp/miniconda.sh -b -p "$CONDA_DIR"
  rm /tmp/miniconda.sh
  echo 'export PATH="$HOME/miniconda3/bin:$PATH"' >> "$HOME/.bashrc"
  export PATH="$HOME/miniconda3/bin:$PATH"
else
  echo "[aws_bootstrap] Miniconda already installed at $CONDA_DIR"
  export PATH="$HOME/miniconda3/bin:$PATH"
fi

echo "[aws_bootstrap] Creating conda env 'k2lab' (if missing)..."
if ! conda env list | grep -q "^k2lab"; then
  conda create -y -n k2lab python=3.10
else
  echo "[aws_bootstrap] Conda env 'k2lab' already exists."
fi

cat <<EOF

[aws_bootstrap] Done.

To start using this environment:

  source "\$HOME/miniconda3/bin/activate"
  conda activate k2lab

Then from inside the k2-lab repo:

  bash step_1_setup_env.sh

EOF
