#!/bin/bash
#SBATCH --job-name=nequix_pft_healthy2d
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

export WANDB_MODE=offline

uv run python -u nequix/pft/train.py configs/nequix-healthy-2d-pft.yml
