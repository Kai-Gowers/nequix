#!/bin/bash
#SBATCH --job-name=nequix_pft_eval
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

# Usage: sbatch eval_healthy2d_finetune.sbatch <path-to-finetuned-checkpoint.nqx> [dataset-dir] [baseline-model] [output-dir] [relax] [start-dir]
# relax: none (default, fixed DFT geometry) | positions | cell  -- template-start end-to-end evals; start-dir defaults to
# workflow/template_structures. Always pass an explicit output-dir for relax runs (e.g. ..._template_relaxpos_eval).
# baseline-model defaults to nequix-omat-1 (project default base model since 2026-09-15); output-dir defaults to the eval script default
# (phonon_comparisons/<dataset-dir>_eval) — pass it explicitly to avoid overwriting an existing eval.
FINETUNED_MODEL="${1:?Usage: sbatch eval_healthy2d_finetune.sbatch <finetuned-checkpoint.nqx> [dataset-dir]}"
DATASET_DIR="${2:-v2_73healthy_split}"
BASELINE_MODEL="${3:-nequix-omat-1}"   # default base since 2026-09-15; pass nequix-mp-1 explicitly for older runs
OUTPUT_DIR="${4:-}"
RELAX="${5:-none}"
START_DIR="${6:-}"

uv run python -u scripts/eval_healthy2d_finetune.py \
  --finetuned-model "$FINETUNED_MODEL" \
  --baseline-model "$BASELINE_MODEL" \
  --dataset-dir "$DATASET_DIR" \
  ${OUTPUT_DIR:+--output-dir "$OUTPUT_DIR"} \
  --relax "$RELAX" \
  ${START_DIR:+--start-dir "$START_DIR"}
