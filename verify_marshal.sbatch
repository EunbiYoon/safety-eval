#!/bin/bash
# W1 verification for MARSHAL-Generalist only (including weight distances). No GPU needed.
# Submit: sbatch verify_marshal.sbatch
#SBATCH -J verify_marshal
#SBATCH -p cpu
#SBATCH -c 4
#SBATCH --mem=32G
#SBATCH -t 02:00:00
#SBATCH -o logs_slurm/%x_%j.out
#SBATCH -e logs_slurm/%x_%j.err
set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./config.sh
hostname; date; module list 2>&1 || true
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-1}"
PY=/scratch/workspace/eunbiyoon_umass_edu-paper/envs/gaze/bin/python   # has huggingface_hub/safetensors/torch
which "$PY" && "$PY" --version
"$PY" verify_checkpoints.py --tags marshal-gen --out results/checkpoint_verification_marshal.json
