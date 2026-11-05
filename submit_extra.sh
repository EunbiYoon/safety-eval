#!/bin/bash
# Submit extra benchmarks: smoke test → (if it passes) model×think generation jobs → per-model scoring jobs
#   bash submit_extra.sh mask   "0"        # MASK, seed 0
#   bash submit_extra.sh xstest "0 1 2 3"  # XSTest, seed 0~3
# If the smoke test fails, the main jobs sit in DependencyNeverSatisfied → fix the cause, scancel, and resubmit.
set -euo pipefail
cd /scratch/workspace/eunbiyoon_umass_edu-paper/safety-eval
BENCH="$1"; SEEDS_ARG="${2:-0}"
N=$(grep -v '^#' models.tsv | grep -cv '^\s*$')
ARRAY="0-$((N - 1))"
COMMON="ALL,BENCHES=$BENCH,MAX_CONNECTIONS=8"

smoke=$(sbatch --parsable -J "x-$BENCH-smoke" --array=0 -t 02:00:00 \
  --export="$COMMON,STAGE=all,LIMIT=5,THINK_MODES=0 1,SEEDS=0,EXTRA_LOG_ROOT=$PWD/logs_extra_smoke" \
  eval_extra.sbatch)
echo "smoke        $smoke"

gens=()
for t in 0 1; do
  g=$(sbatch --parsable -J "x-$BENCH-gen-t$t" --array="$ARRAY" --dependency="afterok:$smoke" \
    --export="$COMMON,STAGE=gen,THINK_MODES=$t,SEEDS=$SEEDS_ARG" eval_extra.sbatch)
  echo "gen think=$t  $g  (array $ARRAY, seeds [$SEEDS_ARG])"; gens+=("$g")
done

s=$(sbatch --parsable -J "x-$BENCH-score" --array="$ARRAY" -p gpu-preempt-highvram,gpu-highvram \
  --dependency="aftercorr:${gens[0]}:${gens[1]}" \
  --export="$COMMON,STAGE=score" eval_extra.sbatch)
echo "score        $s  (starts per model once both think0 and think1 generation finish)"
