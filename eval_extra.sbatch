#!/bin/bash
# XSTest / MASK. One model = one array task (row number in models.tsv).
#   STAGE=gen   : generation only, with the policy server (BENCHES × THINK_MODES × SEEDS)
#   STAGE=score : scoring only, with the judge server (BENCHES)
#   STAGE=all   : generate → score
#   LIMIT=5     : smoke test (give it a separate EXTRA_LOG_ROOT so it doesn't mix with the real results)
#SBATCH -J extra
#SBATCH -p gpu-preempt-highvram
#SBATCH --gres=gpu:l40s:1
#SBATCH --requeue
#SBATCH --exclude=gpu035,ials-gpu035   # server failed to start twice (11/1, 11/3) with "CUDA device busy or unavailable"
#SBATCH -c 8
#SBATCH --mem=64G
#SBATCH -t 12:00:00
#SBATCH -o logs_slurm/%x_%A_%a.out
#SBATCH --mail-user=eunbiyoon@umass.edu
#SBATCH --mail-type=END,FAIL

set -euo pipefail
cd "${SLURM_SUBMIT_DIR}"
source ./config.sh
source ./serve.sh
module load conda/latest 2>/dev/null || true
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV"
hostname; date; which python; python --version

export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-$(cat "$HF_HOME/token" 2>/dev/null || true)}"
STAGE="${STAGE:-all}"
BENCHES="${BENCHES:-xstest mask}"
LIMIT_ARG=""; [[ -n "${LIMIT:-}" ]] && LIMIT_ARG="--limit $LIMIT"

export JOB_LOG_DIR="$PROJ_DIR/logs_slurm/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
mkdir -p "$JOB_LOG_DIR"
mapfile -t ROWS < <(grep -v '^#' models.tsv | grep -v '^\s*$')
IFS=$'\t' read -r TAG REPO GROUP ITER <<< "${ROWS[$SLURM_ARRAY_TASK_ID]}"
GPU="${CUDA_VISIBLE_DEVICES%%,*}"
echo "=== [$SLURM_ARRAY_TASK_ID] $TAG stage=$STAGE benches=[$BENCHES] think=[$THINK_MODES] seeds=[$SEEDS] limit=[${LIMIT:-}] ==="

POLICY_PID=0; JUDGE_PID=0
trap 'stop_server ${POLICY_PID:-0}; stop_server ${JUDGE_PID:-0}' EXIT

if [[ "$STAGE" == gen || "$STAGE" == all ]]; then
  POLICY_PID=$(start_policy "$REPO" "$TAG" "$GPU")
  wait_ready "$POLICY_PORT" "$TAG" "$POLICY_PID"
  for BENCH in $BENCHES; do
    for THINK in $THINK_MODES; do
      for SEED in $SEEDS; do
        # shellcheck disable=SC2086
        python run_extra.py gen --bench "$BENCH" --tag "$TAG" --think "$THINK" --seed "$SEED" $LIMIT_ARG \
          || echo "[FAIL] gen $BENCH $TAG think=$THINK seed=$SEED"
      done
    done
  done
  stop_server "$POLICY_PID"; POLICY_PID=0
fi

if [[ "$STAGE" == score || "$STAGE" == all ]]; then
  JUDGE_PID=$(start_judge "$GPU")
  wait_ready "$JUDGE_PORT" judge "$JUDGE_PID"
  for BENCH in $BENCHES; do
    python run_extra.py score --bench "$BENCH" --tag "$TAG" || echo "[FAIL] score $BENCH $TAG"
  done
fi
echo "=== done $TAG stage=$STAGE ==="
