import argparse
from pathlib import Path

from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import get_model
from inspect_evals.agentharm.benchmark import benign_grading_functions, harmful_grading_functions
from inspect_evals.agentharm.scorer import RefusalJudgeLLM, combined_scorer

JUDGE = "openai-api/judge/judge"
GRADING = {"harmful": harmful_grading_functions, "benign": benign_grading_functions, "chat": None}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--logs", required=True)
    args = ap.parse_args()
    refusal_judge = RefusalJudgeLLM(judge_model=JUDGE)
    semantic_judge = get_model(JUDGE)

    for p in sorted(Path(args.logs).resolve().rglob("*.eval")):
        log = read_eval_log(str(p))
        if log.status != "success" or not log.samples:
            print(f"[skip] {p} status={log.status}"); continue
        if all(s.scores for s in log.samples):
            print(f"[skip] {p} already scored"); continue
        split = (log.eval.metadata or {}).get("split")
        if split not in GRADING:
            print(f"[warn] {p} no split"); continue
        scored = score(log, scorers=combined_scorer(GRADING[split], refusal_judge, semantic_judge),
                       action="overwrite", display="plain")
        write_eval_log(scored, str(p))
        print(f"[ok] scored {p}", flush=True)


if __name__ == "__main__":
    main()
