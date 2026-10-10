#!/bin/bash
#SBATCH --job-name=nequix_eval_cellrelax
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

# Phonon eval with the 16 cellrelax materials' DFT reference read from workflow/FINAL_RESULTS_CELLRELAX
# (scripts/eval_cellrelax_reference.py). Pass the sub-script and its own arguments verbatim:
#   sbatch --job-name=<exp> sbatch/eval_cellrelax_reference.sbatch healthy2d --finetuned-model ... --relax cell --add-d3 both ...
#   sbatch --job-name=<exp> sbatch/eval_cellrelax_reference.sbatch explicit  --finetuned-model ... --output-dir ...
cd "$SLURM_SUBMIT_DIR"
WHICH="${1:?healthy2d|explicit}"; shift

uv run python -u scripts/eval_cellrelax_reference.py "$WHICH" "$@"
