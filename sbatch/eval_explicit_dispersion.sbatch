#!/bin/bash
#SBATCH --job-name=nequix_eval_explicit_d3
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Phonon-band eval with the explicit D3 term added back (explicit_dispersion/README.md).
# Usage: sbatch --job-name=<exp>_eval sbatch/eval_explicit_dispersion.sbatch <finetuned-model> <baseline-model> <add-d3: finetuned|baseline|both|none> <output-dir> [dataset-dir]
cd "$SLURM_SUBMIT_DIR"
FINETUNED="${1:?finetuned model path/alias}"
BASELINE="${2:?baseline model path/alias}"
ADD_D3="${3:?add-d3: finetuned|baseline|both|none}"
OUTPUT_DIR="${4:?output dir}"
DATASET_DIR="${5:-v4_tmd_only}"

uv run python -u explicit_dispersion/eval_explicit_dispersion.py \
  --finetuned-model "$FINETUNED" \
  --baseline-model "$BASELINE" \
  --add-d3 "$ADD_D3" \
  --dataset-dir "$DATASET_DIR" \
  --output-dir "$OUTPUT_DIR"
