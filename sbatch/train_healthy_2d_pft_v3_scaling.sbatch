#!/bin/bash
#SBATCH --job-name=nequix_pft_2h3r_scaling
#SBATCH --output=logs/%x-%j.out
# NOTE: override the job name at submit time so the log is self-describing:
#   sbatch --job-name=<experiment> sbatch/train_healthy_2d_pft_v3_scaling.sbatch ...
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

export WANDB_MODE=offline

# Usage: sbatch train_healthy_2d_pft_v3_scaling.sbatch <config-name-without-.yml>
# e.g. sbatch train_healthy_2d_pft_v3_scaling.sbatch nequix-healthy-2d-pft-v3-50pct
CONFIG="${1:?Usage: sbatch train_healthy_2d_pft_v3_scaling.sbatch <config-name-without-.yml>}"

uv run python -u nequix/pft/train.py "configs/${CONFIG}.yml"
