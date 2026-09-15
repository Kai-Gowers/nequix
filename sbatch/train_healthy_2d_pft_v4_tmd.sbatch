#!/bin/bash
#SBATCH --job-name=nequix_pft_healthy2d_v4_tmd
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

export WANDB_MODE=offline

uv run python -u nequix/pft/train.py configs/nequix-healthy-2d-pft-v4-tmd.yml
