#!/bin/bash
#SBATCH --job-name=mace_finetune_v4_tmd
#SBATCH --output=logs/%x-%j.out
#SBATCH --partition=short
#SBATCH --time=05:00:00
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

cd "$SLURM_SUBMIT_DIR"

VENV="mace_baseline/.venv"
export SSL_CERT_FILE=$("$VENV/bin/python" -c "import certifi; print(certifi.where())")

# Standard (forces-only) MACE fine-tune on the same v4_tmd_only train/val data used
# for Nequix's PFT retrain -- NOT the same recipe (MACE has no Hessian loss, so this
# trains on raw per-displacement DFT forces via --loss forces_only instead). See
# scripts/convert_to_mace_extxyz.py for how the data was built and why.
"$VENV/bin/python" -u -m mace.cli.run_train \
  --name mace_finetune_v4_tmd_only \
  --model_dir mace_baseline/models \
  --checkpoints_dir mace_baseline/checkpoints \
  --log_dir mace_baseline/logs \
  --results_dir mace_baseline/results \
  --foundation_model medium \
  --multiheads_finetuning False \
  --E0s foundation \
  --energy_key energy \
  --forces_key forces \
  --loss forces_only \
  --forces_weight 100.0 \
  --energy_weight 0.0 \
  --compute_stress False \
  --train_file mace_baseline/data/v4_tmd_only_forces/train.xyz \
  --valid_file mace_baseline/data/v4_tmd_only_forces/val.xyz \
  --batch_size 4 \
  --valid_batch_size 4 \
  --max_num_epochs 150 \
  --seed 0 \
  --device cuda \
  --default_dtype float32 \
  --save_cpu
