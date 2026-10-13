#!/bin/bash
#SBATCH --job-name=nequix_twisted
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=06:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch --job-name=<experiment> sbatch/eval_twisted_phonons.sbatch <model.nqx> <output-dir> <structure-dir> [<structure-dir> ...]
# Model-only twisted-bilayer phonons (relax positions with the model, analytical Hessian, Gamma-K-M-Gamma).
# Structures come from workflow/twisted/build_twisted_bilayer.py. Put the largest supercell last.
# Extra options (e.g. --baseline-model nequix-omat-1, --supercell-k 1, --relax none) via TWISTED_EXTRA_ARGS.
# GOTCHA: the login shell sets SBATCH_EXPORT=none, so a plain `TWISTED_EXTRA_ARGS=... sbatch ...` prefix is silently
# dropped (jobs 3145023/3145024 ran with defaults). Pass it explicitly:
#   sbatch --export=ALL,TWISTED_EXTRA_ARGS="--add-d3 finetuned --dft-results ../workflow/FINAL_RESULTS_TWISTED" ...
# and check line 2 of the log ("D3 added to: ...; DFT references: ...").
#
#   sbatch --job-name=twisted_<name>_relaxpos sbatch/eval_twisted_phonons.sbatch   (layout: phonon_comparisons/twisted/README.md) \
#     checkpoints/nequix-healthy-2d-pft-v5-tmd-cellrelax-omat-baresub-e20.nqx phonon_comparisons/twisted/<name>/relaxpos \
#     ../workflow/twisted/structures/MoS2_twist_m{1,5,4,2,3}_near0
MODEL="${1:?Usage: sbatch eval_twisted_phonons.sbatch <model.nqx> <output-dir> <structure-dir>...}"
OUTPUT_DIR="${2:?output-dir required}"
shift 2
[ $# -ge 1 ] || { echo "at least one structure dir required" >&2; exit 2; }

uv run python -u scripts/eval_twisted_phonons.py \
  --model "$MODEL" \
  --output-dir "$OUTPUT_DIR" \
  --structures "$@" \
  ${TWISTED_EXTRA_ARGS:-}
