#!/bin/bash
#   bash run_local.sh smoke      # Qwen3-8B, 5 samples
#   bash run_local.sh            # all of models.tsv
#   bash run_local.sh 1 3        # only models 1 and 3
#   RUN_SEEDS="0 1 2 3" bash run_local.sh
set -uo pipefail
cd /scratch/workspace/eunbiyoon_umass_edu-paper/safety-eval

module load conda/latest 2>/dev/null || true
eval "$(conda shell.bash hook)"
conda activate safety-eval

source ./config.sh
source ./serve.sh
SEEDS="${RUN_SEEDS:-0}"

export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
GPU="${CUDA_VISIBLE_DEVICES%%,*}"
nvidia-smi -L || { echo "No GPU visible — check that you are running on a compute node"; exit 1; }
[[ "$JUDGE_REPO" == *CHANGE_ME* || "$JUDGE_REPO" == *"<"* || "$JUDGE_REVISION" == *"<"* ]] && { echo "Fill in JUDGE_REPO in config.sh first"; exit 1; }
echo "HF_HOME=$HF_HOME | SEEDS=$SEEDS | GPU=$GPU"

mkdir -p logs_slurm
mapfile -t ROWS < <(grep -v '^#' models.tsv | grep -v '^\s*$')

SMOKE=0
if [[ "${1:-}" == "smoke" ]]; then
  SMOKE=1; SEEDS="0"; IDX=(0)
elif [[ $# -gt 0 ]]; then
  IDX=("$@")
else
  IDX=($(seq 0 $((${#ROWS[@]} - 1))))
fi
LIMIT_ARG=""; [[ $SMOKE == 1 ]] && LIMIT_ARG="--limit 5"

POLICY_PID=0; JUDGE_PID=0
trap 'stop_server $POLICY_PID; stop_server $JUDGE_PID' EXIT

for i in "${IDX[@]}"; do
  IFS=$'\t' read -r TAG REPO GROUP ITER <<< "${ROWS[$i]}"
  export JOB_LOG_DIR="$PROJ_DIR/logs_slurm/local_${TAG}"
  mkdir -p "$JOB_LOG_DIR"
  echo; echo "================ [$i] $TAG ($REPO) ================"

  POLICY_PID=$(start_policy "$REPO" "$TAG" "$GPU")
  if ! wait_ready "$POLICY_PORT" "$TAG" "$POLICY_PID"; then
    tail -30 "$JOB_LOG_DIR/vllm_policy_${TAG}.log"
    stop_server $POLICY_PID; POLICY_PID=0; continue
  fi
  for THINK in $THINK_MODES; do
    for SEED in $SEEDS; do
      python run_eval.py --tag "$TAG" --think "$THINK" --seed "$SEED" --no-score $LIMIT_ARG \
        || echo "[FAIL] gen $TAG think=$THINK seed=$SEED"
    done
  done
  stop_server $POLICY_PID; POLICY_PID=0

  JUDGE_PID=$(start_judge "$GPU")
  if ! wait_ready "$JUDGE_PORT" judge "$JUDGE_PID"; then
    tail -30 "$JOB_LOG_DIR/vllm_judge.log"
    stop_server $JUDGE_PID; JUDGE_PID=0; continue
  fi
  python score_logs.py --logs "$LOG_ROOT/$TAG"
  python rescore_visible.py --logs "$LOG_ROOT/$TAG"
  stop_server $JUDGE_PID; JUDGE_PID=0

  echo "================ done $TAG ================"
done

if [[ $SMOKE == 1 ]]; then
  echo; echo "If the smoke test passes, clear results with: rm -rf logs/$(cut -f1 <<< "${ROWS[0]}") results/visible_refusal results/*.csv"
else
  python analyze.py
fi
