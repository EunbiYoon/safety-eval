#!/bin/bash
# ============================================================
# One model = one array task. The judge is loaded only once per job;
# all of think × seed × split are run, followed by visible-only rescoring.
#
# Full run:          sbatch --array=0-11 eval_array.sbatch      (match the number of rows in models.tsv)
# Smoke test:        SMOKE=1 sbatch --array=0 eval_array.sbatch  (qwen3-8b, 5 samples)
# Concurrency limit: --array=0-11%2   (adjust %N to the L40S quota)
# ============================================================
#SBATCH -J agentharm
#SBATCH -p gpu                      # [EDIT] Unity partition (gpu / gpu-preempt). For preempt, skip logic handles requeues
#SBATCH --constraint=l40s           # [EDIT] check the Unity GPU-type constraint name
#SBATCH -G 2                        # GPU0 = policy 8B, GPU1 = judge 27B FP8  (3 for a bf16 judge)
#SBATCH -c 16
#SBATCH --mem=128G
#SBATCH -t 24:00:00
#SBATCH -o logs_slurm/%x_%A_%a.out

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./config.sh
source ./serve.sh
module load conda/latest 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV"

mkdir -p logs_slurm
export JOB_LOG_DIR="$PROJ_DIR/logs_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_LOG_DIR"

# --- model for this array index ---
mapfile -t ROWS < <(grep -v '^#' models.tsv | grep -v '^\s*$')
IFS=$'\t' read -r TAG REPO GROUP ITER <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
echo "=== [$SLURM_ARRAY_TASK_ID] $TAG ($REPO) group=$GROUP iter=$ITER ==="

# Use the physical GPU ids SLURM assigned (hardcoding 0,1 could grab someone else's GPU)
IFS=',' read -r -a GPUS <<< "$CUDA_VISIBLE_DEVICES"
POLICY_GPU="${GPUS[0]}"
JUDGE_GPUS=$(IFS=,; echo "${GPUS[*]:1}")

trap 'stop_server ${POLICY_PID:-0}; stop_server ${JUDGE_PID:-0}' EXIT

JUDGE_PID=$(start_judge "$JUDGE_GPUS")
POLICY_PID=$(start_policy "$REPO" "$TAG" "$POLICY_GPU")
wait_ready "$JUDGE_PORT" judge "$JUDGE_PID"
wait_ready "$POLICY_PORT" "$TAG" "$POLICY_PID"

if [[ "${SMOKE:-0}" == "1" ]]; then
  SEEDS="0"; LIMIT_ARG="--limit 5"
else
  LIMIT_ARG=""
fi

for THINK in $THINK_MODES; do
  for SEED in $SEEDS; do
    # shellcheck disable=SC2086
    python run_eval.py --tag "$TAG" --think "$THINK" --seed "$SEED" $LIMIT_ARG \
      || echo "[FAIL] $TAG think=$THINK seed=$SEED — continuing"
  done
done

# Rescore without reasoning (while the judge is still up)
python rescore_visible.py --logs "$LOG_ROOT/$TAG"

echo "=== done $TAG ==="
