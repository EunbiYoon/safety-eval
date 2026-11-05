# AgentHarm safety eval — game-RL Qwen3-8B checkpoints

Unity(SLURM) + L40S + vLLM + Inspect AgentHarm.

## Run order

```bash
# 0. Setup: the [EDIT] items in config.sh (HF_HOME, JUDGE_REPO/REVISION, partition)
bash setup_env.sh                                  # login node, once

# 1. ⚠️ W1 required — don't use GPUs until this passes
srun -p cpu -c 8 --mem=64G -t 2:00:00 python verify_checkpoints.py
#    - whether MARSHAL is really 8B (arch_ok)
#    - whether SPIRAL starts from Qwen3-8B or from Base (init_verdict)
#    - whether each template supports tools / enable_thinking

# 2. Smoke test (Qwen3-8B, 5 samples, greedy)
SMOKE=1 sbatch --array=0 eval_array.sbatch
#    check logs_slurm/ → delete the logs, then do the real run:  rm -rf logs/qwen3-8b

# 3. Full sweep (12 rows in models.tsv → 0-11, %2 = 2 concurrent jobs)
sbatch --array=0-11%2 eval_array.sbatch

# 4. Analysis (no GPU needed)
python analyze.py                  # if SPIRAL starts from Base: --ref qwen3-8b-base

# 5. Judge validation
python judge_validation.py sample --n 150   # fill in the human_refusal column by hand
python judge_validation.py kappa
```

## Design decisions and rationale

| Decision | Reason |
|---|---|
| No Ollama, vLLM only | GGUF quantization + different tool parsing = confound |
| Force `enable_thinking=false` on the judge server | If the judge puts its answer inside `<think>`, `Answer:` parsing fails → missed refusals |
| `rescore_visible.py` | The original scorer passes the reasoning to the judge too → can inflate refusal rates for think-on models. Report both values |
| think-on `max_tokens=12288` | Truncation at 4096 means no tool call → miscounted as 'safe' |
| seed 0 = greedy, 1–3 = sampling | Comparable with the benchmark default (greedy) + variance estimates |
| Behavior-level cluster bootstrap | The 4 variants of the same behavior are not independent |
| `parse_fail`, `truncated` alarms | Watch for failed tool calls looking like safety |

## Outputs
- `results/checkpoint_verification.json`
- `results/samples.csv` · `summary.csv` · `delta_vs_ref.csv`
- `results/spiral_curve.png`
- `results/human_labels.csv`
# safety-eval
