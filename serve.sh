#!/bin/bash
# vLLM server helpers. Source this inside an sbatch script.
#   start_policy <hf_repo> <tag> <gpu_id>
#   start_judge  <gpu_ids>            (e.g. "1" or "1,2")
#   wait_ready   <port> <name> [pid]   (with pid, fail immediately if the server dies)
#   stop_server  <pid>

start_policy() {
  local repo="$1" tag="$2" gpu="$3"
  CUDA_VISIBLE_DEVICES="$gpu" vllm serve "$repo" \
    --served-model-name "$tag" \
    --port "$POLICY_PORT" \
    --dtype bfloat16 \
    --max-model-len "$POLICY_MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --enable-auto-tool-choice \
    --tool-call-parser "$TOOL_PARSER" \
    --reasoning-parser "$REASONING_PARSER" \
    --seed 0 \
    > "$JOB_LOG_DIR/vllm_policy_${tag}.log" 2>&1 &
  echo $!
}

start_judge() {
  local gpus="$1"
  local quant_args=()
  [[ -n "$JUDGE_QUANT" ]] && quant_args=(--quantization "$JUDGE_QUANT")
  # Force enable_thinking=false as the server default so the judge can never hide its
  # answer in <think> and break the "Answer:" parsing.
  # shellcheck disable=SC2086
  CUDA_VISIBLE_DEVICES="$gpus" vllm serve "$JUDGE_REPO" \
    --revision "$JUDGE_REVISION" \
    --served-model-name judge \
    --port "$JUDGE_PORT" \
    --max-model-len "$JUDGE_MAX_LEN" \
    --gpu-memory-utilization 0.90 \
    --tensor-parallel-size "$JUDGE_TP" \
    "${quant_args[@]}" \
    --default-chat-template-kwargs '{"enable_thinking": false}' \
    --seed 0 \
    $JUDGE_EXTRA_ARGS \
    > "$JOB_LOG_DIR/vllm_judge.log" 2>&1 &
  echo $!
}

wait_ready() {
  local port="$1" name="$2" pid="${3:-}" t=0
  # Also check that the responding server is serving this model (served-model-name), to avoid attaching to another job's server
  until curl -sf "http://127.0.0.1:${port}/v1/models" 2>/dev/null | grep -q "\"id\": *\"${name}\""; do
    # If a pid is given, fail right away when the server process dies instead of waiting 30 minutes
    if [[ -n "$pid" ]] && ! kill -0 "$pid" 2>/dev/null; then
      echo "[ERROR] $name server process ($pid) died (${t}s). Check logs: $JOB_LOG_DIR" >&2
      return 1
    fi
    sleep 10; t=$((t + 10))
    if (( t > 1800 )); then
      echo "[ERROR] $name server did not come up within 30 minutes. Check logs: $JOB_LOG_DIR" >&2
      return 1
    fi
  done
  echo "[ok] $name ready on :$port (${t}s)"
}

stop_server() {
  local pid="$1"
  # with pid=0, kill 0 = SIGTERM to the whole process group → the job ends as FAILED(0:15)
  [[ -z "$pid" || "$pid" == 0 ]] && return 0
  kill "$pid" 2>/dev/null || true
  wait "$pid" 2>/dev/null || true
  sleep 5   # wait for GPU memory to be released
}
