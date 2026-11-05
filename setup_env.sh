#!/bin/bash
# Run once on a login node: bash setup_env.sh
# (installing vLLM downloads CUDA builds, so it takes a while)
set -euo pipefail
source "$(dirname "$0")/config.sh"

module load conda/latest 2>/dev/null || true   # [EDIT] check the Unity module name: module avail conda

conda create -y -n "$CONDA_ENV" python=3.11
conda activate "$CONDA_ENV" 2>/dev/null || source activate "$CONDA_ENV"

pip install -U pip
pip install vllm                         # serving
pip install inspect_ai inspect_evals     # eval harness (includes AgentHarm)
pip install huggingface_hub safetensors torch pandas numpy matplotlib openai

mkdir -p "$HF_HOME" "$LOG_ROOT" "$RESULTS_DIR"

# record pinned versions (for the paper's reproducibility section)
pip freeze > "$PROJ_DIR/requirements.lock.txt"
python -c "import vllm, inspect_ai; print('vllm', vllm.__version__, '| inspect_ai', inspect_ai.__version__)"

echo "If you need an HF token: huggingface-cli login (AgentHarm is not gated)"
