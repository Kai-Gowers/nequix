#!/bin/bash
#SBATCH --job-name=nequix_gamma_interlayer
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch eval_gamma_interlayer_modes.sbatch <path-to-finetuned-checkpoint.nqx> [dataset-dir] [output-dir]
FINETUNED_MODEL="${1:?Usage: sbatch eval_gamma_interlayer_modes.sbatch <finetuned-checkpoint.nqx> [dataset-dir] [output-dir]}"
DATASET_DIR="${2:-v3_2h3r}"
OUTPUT_DIR="${3:-}"

uv run python -u scripts/eval_gamma_interlayer_modes.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model nequix-mp-1 \
  --dataset-dir "$DATASET_DIR" \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}
