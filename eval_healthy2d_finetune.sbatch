#!/bin/bash
#SBATCH --job-name=nequix_pft_v2_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch eval_healthy2d_finetune.sbatch <path-to-finetuned-checkpoint.nqx>
FINETUNED_MODEL="${1:?Usage: sbatch eval_healthy2d_finetune.sbatch <finetuned-checkpoint.nqx>}"

uv run python -u scripts/eval_healthy2d_finetune.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model nequix-mp-1
