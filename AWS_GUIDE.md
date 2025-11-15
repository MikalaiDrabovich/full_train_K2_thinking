# AWS Guide – Running K2 Lab on an 8×H100 Instance

This document gives you a pragmatic way to run the full K2 Lab pipeline
on an AWS instance with 8×H100 GPUs.

## 1. Instance type & AMI assumptions

These scripts assume:

- You are using an instance type with **8×H100**, e.g. `p5.48xlarge`.
- You are running either:
  - an **AWS Deep Learning AMI (DLAMI)** with PyTorch, or
  - a recent Ubuntu-based image where you will install CUDA / drivers yourself.

The DLAMI route is strongly recommended because:
- NVIDIA drivers and CUDA/cuDNN are already configured.
- Conda is preinstalled and tested with GPUs.

## 2. Bring up the instance (high-level)

1. In the AWS console (or via `aws ec2` CLI):
   - Launch a `p5.48xlarge` (or similar).
   - Attach an EBS volume with enough space for:
     - model checkpoints (hundreds of GB if you keep multiple versions),
     - your datasets.

2. SSH into the instance:

   ```bash
   ssh -i /path/to/your-key.pem ubuntu@EC2_PUBLIC_DNS
   ```

3. (Optional but recommended) Attach and mount a dedicated EBS volume for
   `/data` and put your `full_train_K2_thinking` repo + datasets there.

## 3. Repo & pipeline orchestration scripts

We add two root-level AWS scripts:

- `aws_step_0_bootstrap_ubuntu.sh`
  - Installs system packages, git, tmux, and optionally miniconda.
  - Intended for a *vanilla* Ubuntu image. On a DLAMI, many steps will be
    no-ops or already done.

- `aws_step_1_run_full_pipeline_tmux.sh`
  - Assumes you already have the repo on the instance.
  - Creates a `tmux` session and runs the full K2 Lab pipeline inside it
    (steps 1 through 9).
  - You can detach (`Ctrl-b d`) and reattach later.

## 4. Typical usage on AWS

1. Copy the repo (or zip) onto the instance (from your laptop):

   ```bash
   scp -i /path/to/key.pem full_train_K2_thinking_aws.zip ubuntu@EC2_PUBLIC_DNS:~
   ```

2. On the instance:

   ```bash
   unzip full_train_K2_thinking_aws.zip
   cd full_train_K2_thinking
   ```

3. (Optional) Bootstrap system deps if you’re not on a DLAMI:

   ```bash
   bash aws_step_0_bootstrap_ubuntu.sh
   ```

   - This is idempotent-ish: re-running it usually won’t break things,
     but it’s meant to be run once when you first bring up the instance.

4. Configure the repo:

   - Edit `configs/env.yaml`:
     - Set your Hugging Face token (if needed).
     - Check model repo IDs.
     - Adjust paths if you want to use `/data` instead of `./`.

5. Launch the full pipeline in a tmux session:

   ```bash
   bash aws_step_1_run_full_pipeline_tmux.sh
   ```

   - This will:
     - Create a `tmux` session named `k2lab`.
     - Inside that session, sequentially run:
       - `step_1_setup_env.sh`
       - `step_2_download_k2.sh`
       - `step_3_prepare_corpora.sh`
       - `step_4_continual_pretrain.sh`
       - `step_5_domain_sft.sh`
       - `step_6_reasoning_sft.sh`
       - `step_7_dpo_reasoning.sh`
       - `step_8_distill_student.sh`
       - `step_9_evaluate.sh`

6. Monitor progress:

   ```bash
   tmux attach -t k2lab
   ```

   - Use `Ctrl-b d` to detach again without stopping the run.

## 5. Customizing for your environment

- If you want to skip heavy stages (e.g. DPO at first), edit
  `aws_step_1_run_full_pipeline_tmux.sh` and comment out some steps.
- If you store data or models under `/data`, set:

  ```bash
  export K2LAB_REPO_DIR=/data/full_train_K2_thinking
  ```

  before running the AWS scripts, or change the `REPO_DIR` default in the
  script itself.
