#!/bin/bash
#SBATCH --job-name=nequix_interlayer_zoom
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=01:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch plot_interlayer_zoom.sbatch <path-to-finetuned-checkpoint.nqx>
FINETUNED_MODEL="${1:?Usage: sbatch plot_interlayer_zoom.sbatch <finetuned-checkpoint.nqx>}"

uv run python -u scripts/plot_interlayer_zoom.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model nequix-mp-1 \
  --materials MoS2_WS2_3R MoS2_WSe2_3R MoS2_bilayer_2H MoS2_bilayer_3R WSe2_TiS2_3R WSe2_WTe2_3R
