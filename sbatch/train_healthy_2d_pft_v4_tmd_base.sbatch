#!/bin/bash
#SBATCH --job-name=nequix_pft_v4_tmd_base
#SBATCH --output=logs/%x-%j.out
# NOTE: override the job name at submit time so the log is self-describing:
#   sbatch --job-name=<experiment> sbatch/train_healthy_2d_pft_v4_tmd_base.sbatch ...
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Base-model ablation on the v4 TMD-only split: same recipe as
# train_healthy_2d_pft_v4_tmd.sbatch, config passed as $1.
# Usage: sbatch sbatch/train_healthy_2d_pft_v4_tmd_base.sbatch configs/nequix-healthy-2d-pft-v4-tmd-<oam|omat>.yml
cd "$SLURM_SUBMIT_DIR"
CONFIG="${1:?Usage: sbatch train_healthy_2d_pft_v4_tmd_base.sbatch <config.yml>}"

export WANDB_MODE=offline

uv run python -u nequix/pft/train.py "$CONFIG"
