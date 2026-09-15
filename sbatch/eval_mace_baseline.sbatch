#!/bin/bash
#SBATCH --job-name=eval_mace_baseline
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

cd "$SLURM_SUBMIT_DIR"

VENV="mace_baseline/.venv"
export SSL_CERT_FILE=$("$VENV/bin/python" -c "import certifi; print(certifi.where())")

"$VENV/bin/python" -u scripts/eval_mace_baseline.py \
  --model medium \
  --device cpu \
  --output-dir phonon_comparisons/mace_mp_baseline
