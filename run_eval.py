import argparse
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai.log import read_eval_log
from inspect_ai.model import get_model
from inspect_evals.agentharm import agentharm, agentharm_benign

JUDGE = "openai-api/judge/judge"
SAMPLING = {1: dict(temperature=0.6, top_p=0.95), 0: dict(temperature=0.7, top_p=0.8)}


def build_task(split):
    common = dict(split="test_public", refusal_judge=JUDGE, semantic_judge=JUDGE)
    if split == "harmful":
        return agentharm(**common)
    if split == "benign":
        return agentharm_benign(**common)
    if split == "chat":
        return agentharm(chat_dataset=True, **common)
    raise ValueError(split)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tag", required=True)
    ap.add_argument("--think", type=int, choices=[0, 1], required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--splits", default=os.environ.get("SPLITS", "harmful benign chat"))
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--no-score", action="store_true")
    args = ap.parse_args()

    max_tokens = int(os.environ["MAX_TOKENS_THINK" if args.think else "MAX_TOKENS_NOTHINK"])
    # With spiral-i383 think=1, long reasoning requests hit the default timeout and 7% were dropped.
    # This only raises the limit, so results for samples that finished generating are unchanged.
    gen = dict(seed=args.seed, max_tokens=max_tokens,
               attempt_timeout=1800, timeout=3600,
               extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.think)}})
    gen.update(temperature=0.0) if args.seed == 0 else gen.update(SAMPLING[args.think])

    policy = get_model(f"openai-api/policy/{args.tag}")

    for split in args.splits.split():
        log_dir = Path(os.environ["LOG_ROOT"]) / args.tag / f"think{args.think}" / f"seed{args.seed}" / split
        done = False
        for p in log_dir.glob("*.eval"):
            if read_eval_log(str(p), header_only=True).status == "success":
                done = True
            else:
                print(f"[clean] removing incomplete log: {p}")
                p.unlink()
        if done:
            print(f"[skip] {log_dir} already done")
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] {args.tag} think={args.think} seed={args.seed} split={split}", flush=True)
        inspect_eval(
            build_task(split), model=policy, log_dir=str(log_dir), log_format="eval",
            limit=args.limit, max_connections=int(os.environ.get("MAX_CONNECTIONS", "32")), fail_on_error=0.05,
            metadata=dict(tag=args.tag, think=args.think, seed=args.seed, split=split),
            score=not args.no_score, **gen,
        )


if __name__ == "__main__":
    main()
