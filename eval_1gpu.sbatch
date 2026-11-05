#!/bin/bash
#SBATCH -J agentharm1g
#SBATCH -p gpu-preempt-highvram
#SBATCH --gres=gpu:l40s:1
#SBATCH --requeue
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 24:00:00
#SBATCH -o logs_slurm/%x_%A_%a.out
#SBATCH --mail-user=eunbiyoon@umass.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./config.sh
source ./serve.sh
module load conda/latest 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV"

export JOB_LOG_DIR="$PROJ_DIR/logs_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_LOG_DIR"

mapfile -t ROWS < <(grep -v '^#' models.tsv | grep -v '^\s*$')
IFS=$'\t' read -r TAG REPO GROUP ITER <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
echo "=== [$SLURM_ARRAY_TASK_ID] $TAG ($REPO) — 1 GPU ==="
GPU="${CUDA_VISIBLE_DEVICES%%,*}"

if [[ "${SMOKE:-0}" == "1" ]]; then SEEDS="0"; LIMIT_ARG="--limit 5"; else LIMIT_ARG=""; fi

# Stage 1: generate
POLICY_PID=$(start_policy "$REPO" "$TAG" "$GPU")
trap 'stop_server ${POLICY_PID:-0}; stop_server ${JUDGE_PID:-0}' EXIT
wait_ready "$POLICY_PORT" "$TAG" "$POLICY_PID"
for THINK in $THINK_MODES; do
  for SEED in $SEEDS; do
    python run_eval.py --tag "$TAG" --think "$THINK" --seed "$SEED" --no-score $LIMIT_ARG \
      || echo "[FAIL] gen $TAG think=$THINK seed=$SEED"
  done
done
stop_server "$POLICY_PID"; POLICY_PID=0

# GEN_ONLY=1: for generation jobs split up per split. Scoring is done all at once by the last job
# (files can get corrupted if several jobs score/overwrite the same log at the same time)
if [[ "${GEN_ONLY:-0}" == "1" ]]; then echo "=== gen-only done $TAG ($THINK_MODES / $SEEDS / $SPLITS) ==="; exit 0; fi

# Stage 2: score
JUDGE_PID=$(start_judge "$GPU")
wait_ready "$JUDGE_PORT" judge "$JUDGE_PID"
python score_logs.py --logs "$LOG_ROOT/$TAG"
python rescore_visible.py --logs "$LOG_ROOT/$TAG"
echo "=== done $TAG ==="
