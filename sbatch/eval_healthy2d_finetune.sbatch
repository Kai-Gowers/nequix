#!/bin/bash
#SBATCH --job-name=nequix_pft_eval
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch eval_healthy2d_finetune.sbatch <path-to-finetuned-checkpoint.nqx> [dataset-dir] [baseline-model] [output-dir]
# baseline-model defaults to nequix-mp-1; output-dir defaults to the eval script default
# (phonon_comparisons/<dataset-dir>_eval) — pass it explicitly to avoid overwriting an existing eval.
FINETUNED_MODEL="${1:?Usage: sbatch eval_healthy2d_finetune.sbatch <finetuned-checkpoint.nqx> [dataset-dir]}"
DATASET_DIR="${2:-v2_73healthy_split}"
BASELINE_MODEL="${3:-nequix-mp-1}"
OUTPUT_DIR="${4:-}"

uv run python -u scripts/eval_healthy2d_finetune.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model "$BASELINE_MODEL" \
  --dataset-dir "$DATASET_DIR" \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"}
