#!/bin/bash
#SBATCH --job-name=nequix_pft_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch eval_healthy2d_finetune.sbatch <path-to-finetuned-checkpoint.nqx> [dataset-dir]
FINETUNED_MODEL="${1:?Usage: sbatch eval_healthy2d_finetune.sbatch <finetuned-checkpoint.nqx> [dataset-dir]}"
DATASET_DIR="${2:-v2_73healthy_split}"

uv run python -u scripts/eval_healthy2d_finetune.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model nequix-mp-1 \
  --dataset-dir "$DATASET_DIR"
