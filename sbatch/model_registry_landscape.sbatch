#!/bin/bash
#SBATCH --job-name=nequix_landscape
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch --job-name=landscape_<tag> sbatch/model_registry_landscape.sbatch <model.nqx> <output-dir> [--pilot|--all|--materials ...]
# Model-side registry x gap landscape (scripts/model_registry_landscape.py); baseline omat-1 zero-shot by default.
MODEL="${1:?Usage: sbatch model_registry_landscape.sbatch <model.nqx> <output-dir> [--pilot|--all|--materials ...]}"
OUTPUT_DIR="${2:?output-dir required}"
shift 2

uv run python -u scripts/model_registry_landscape.py --model "$MODEL" --output-dir "$OUTPUT_DIR" "$@"
