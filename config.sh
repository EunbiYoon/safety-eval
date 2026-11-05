# ============================================================
# config.sh — single config file sourced by every script
# For Unity, only the values marked [EDIT] below need changing.
# ============================================================

# [EDIT] Project root (where this folder lives on Unity)
export PROJ_DIR="${PROJ_DIR:-/scratch/workspace/eunbiyoon_umass_edu-paper/safety-eval}"

# [EDIT] HF cache: the home directory quota is small, so use /work or /scratch.
#   10 x 8B models ≈ 160GB + 27B judge ≈ 55GB. Pick somewhere with room.
export HF_HOME="${HF_HOME:-/scratch/workspace/eunbiyoon_umass_edu-paper/hf_cache}"

# [EDIT] conda environment name
export CONDA_ENV="safety-eval"

# ---------- Serving ----------
# When two jobs land on the same node they share the fixed port (8000) and requests go to the other job's server
# (spiral-blend attached to the qwen3-8b server and everything failed with 404). Separate ports by job ID.
# The job-ID-based calculation actually gave every job 20000 and failed → ask the OS for a free port instead.
# (Existing env vars are ignored: sbatch may pass down the submitting shell's values as-is)
_free_port() { python3 -c 'import socket; s=socket.socket(); s.bind(("127.0.0.1",0)); print(s.getsockname()[1])'; }
export POLICY_PORT="$(_free_port)"
export JUDGE_PORT="$(_free_port)"
while [[ "$JUDGE_PORT" == "$POLICY_PORT" ]]; do JUDGE_PORT="$(_free_port)"; done
# The flashinfer sampler JIT kernel gets built against system CUDA 12 and looks for libcudart.so.12,
# which kills the engine (the env only has the cu13 runtime). Use the PyTorch sampler instead.
export VLLM_USE_FLASHINFER_SAMPLER=0

# vLLM parsers (shared across the Qwen3 family)
export TOOL_PARSER="hermes"        # Qwen3's <tool_call> format
export REASONING_PARSER="qwen3"     # splits <think> into reasoning_content

export POLICY_MAX_LEN=32768
export JUDGE_MAX_LEN=32768   # raised because rescore_visible failed on a 16385-token prompt

# ---------- Judge ----------
# Use the official instruct model (no community uncensored variants). Pin the revision hash too (reproducibility).
# It is a different family from the policy (Qwen3), which also reduces self-preference bias.
export JUDGE_REPO="google/gemma-3-27b-it"
export JUDGE_REVISION="005ad3404e59d6023443cb575daa05336842228a"
# FP8 = one L40S. For bf16, set JUDGE_QUANT="" and JUDGE_TP=2, and --gpus=3 in the sbatch.
export JUDGE_QUANT="fp8"
export JUDGE_TP=1
# Gemma3 is a multimodal architecture (Gemma3ForConditionalGeneration) → disable image input to save memory
export JUDGE_EXTRA_ARGS='--limit-mm-per-prompt {"image":0}'

# ---------- Generation settings ----------
# seed 0 = greedy (benchmark default, temperature 0)
# seed 1..N = sampling (Qwen3 recommended values). For variance/CI estimates.
export SEEDS="${SEEDS:-0 1 2 3}"
export THINK_MODES="${THINK_MODES:-0 1}"  # 0 = thinking off, 1 = thinking on
export SPLITS="${SPLITS:-harmful benign chat}"
export MAX_TOKENS_NOTHINK=4096      # benchmark default
export MAX_TOKENS_THINK=12288       # avoids the illusion where thinking cut off at 4096 is counted as 'no action'

export LOG_ROOT="$PROJ_DIR/logs"
export RESULTS_DIR="$PROJ_DIR/results"

# dummy keys (vLLM doesn't check keys; Inspect only requires the variables to exist)
export POLICY_API_KEY="EMPTY"
export JUDGE_API_KEY="EMPTY"
export POLICY_BASE_URL="http://127.0.0.1:${POLICY_PORT}/v1"
export JUDGE_BASE_URL="http://127.0.0.1:${JUDGE_PORT}/v1"

# So Inspect doesn't die looking for an OpenAI key (its default judge is gpt-4o)
export OPENAI_API_KEY="${OPENAI_API_KEY:-unused}"
