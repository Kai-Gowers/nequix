#!/bin/bash
#SBATCH --job-name=nequix_gamma_interlayer
#SBATCH --output=logs/%x-%j.out
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

# Extra options (e.g. --add-d3 finetuned --all-bilayers) via GAMMA_EXTRA_ARGS; baseline via GAMMA_BASELINE
# (default nequix-mp-1 kept for the old runs; pass models/nequix-omat-1.nqx for the omat-era models).
# GOTCHA: SBATCH_EXPORT=none on the login node -> pass with --export=ALL,GAMMA_EXTRA_ARGS="...",GAMMA_BASELINE=...
uv run python -u scripts/eval_gamma_interlayer_modes.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model "${GAMMA_BASELINE:-nequix-mp-1}" \
  --dataset-dir "$DATASET_DIR" \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
  ${GAMMA_EXTRA_ARGS:-}
