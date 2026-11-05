# AgentHarm safety eval — game-RL Qwen3-8B checkpoints

Unity(SLURM) + L40S + vLLM + Inspect AgentHarm.

## 실행 순서

```bash
# 0. 설정: config.sh 의 [EDIT] 항목 (HF_HOME, JUDGE_REPO/REVISION, 파티션)
bash setup_env.sh                                  # 로그인 노드, 1회

# 1. ⚠️ W1 필수 — 이게 통과하기 전엔 GPU 쓰지 말 것
srun -p cpu -c 8 --mem=64G -t 2:00:00 python verify_checkpoints.py
#    - MARSHAL이 진짜 8B인지 (arch_ok)
#    - SPIRAL이 Qwen3-8B 출발인지 Base 출발인지 (init_verdict)
#    - 각 템플릿이 tools / enable_thinking 지원하는지

# 2. 스모크 테스트 (Qwen3-8B, 5샘플, greedy)
SMOKE=1 sbatch --array=0 eval_array.sbatch
#    logs_slurm/ 확인 → 로그 지우고 본 실행:  rm -rf logs/qwen3-8b

# 3. 전체 스윕 (models.tsv 12행 → 0-11, %2 = 동시 2잡)
sbatch --array=0-11%2 eval_array.sbatch

# 4. 분석 (GPU 불필요)
python analyze.py                  # SPIRAL이 Base 출발이면 --ref qwen3-8b-base

# 5. Judge 검증
python judge_validation.py sample --n 150   # human_refusal 열 직접 채우기
python judge_validation.py kappa
```

## 설계상 결정과 근거

| 결정 | 이유 |
|---|---|
| Ollama 제외, vLLM만 | GGUF 양자화 + 다른 툴 파싱 = confound |
| judge 서버에 `enable_thinking=false` 강제 | judge가 답을 `<think>` 안에 넣으면 `Answer:` 파싱 실패 → 거절 누락 |
| `rescore_visible.py` | 원본 scorer는 reasoning까지 judge에 넘김 → think-on 모델의 거절률 부풀림 가능. 두 값 모두 보고 |
| think-on `max_tokens=12288` | 4096에서 잘리면 툴 호출 없음 → '안전'으로 오집계 |
| seed 0 = greedy, 1–3 = 샘플링 | 벤치마크 기본(greedy)과 비교 가능 + 분산 추정 |
| behavior 단위 cluster bootstrap | 같은 behavior의 4개 변형은 독립 아님 |
| `parse_fail`, `truncated` 경보 | 툴 호출 실패가 안전처럼 보이는 착시 감시 |

## 출력
- `results/checkpoint_verification.json`
- `results/samples.csv` · `summary.csv` · `delta_vs_ref.csv`
- `results/spiral_curve.png`
- `results/human_labels.csv`
# safety-eval
