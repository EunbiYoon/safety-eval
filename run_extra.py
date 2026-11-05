"""
Generation/scoring for additional safety benchmarks beyond AgentHarm (XSTest, MASK).

  gen:   python run_extra.py gen   --bench xstest --tag qwen3-8b --think 0 --seed 0
  score: python run_extra.py score --bench xstest --tag qwen3-8b      (requires judge server)

Logs: $EXTRA_LOG_ROOT/<bench>/<tag>/think{t}/seed{s}/<part>/*.eval
  xstest part = safe | unsafe   (safe = measures over-refusal, unsafe = measures correct refusal)
  mask   part = all

Setup notes (for the paper's methods section):
  - judge is the same Gemma-3-27B-it as AgentHarm (openai-api/judge/judge). Original defaults are
    XSTest=gpt-4o, MASK=gpt-4o(binary)/o3-mini(numeric).
  - MASK judge temperature is 0.0 instead of the original default 1.0 (reproducibility). Other MASK settings are original defaults.
  - MASK generation uses original defaults (temperature 1.0, max_tokens 1000). Only for think=1 is
    max_tokens raised to MAX_TOKENS_THINK so reasoning isn't truncated.
  - XSTest generation follows the same rule as AgentHarm: seed 0 = greedy, seed 1.. = Qwen3 recommended sampling.
"""
import argparse
import os
from pathlib import Path

from inspect_ai import eval as inspect_eval
from inspect_ai import score
from inspect_ai.log import read_eval_log, write_eval_log
from inspect_ai.model import get_model

JUDGE = os.environ.get("EXTRA_JUDGE", "openai-api/judge/judge")   # for testing: EXTRA_JUDGE=mockllm/model
SAMPLING = {1: dict(temperature=0.6, top_p=0.95), 0: dict(temperature=0.7, top_p=0.8)}
PARTS = {"xstest": ["safe", "unsafe"], "mask": ["all"]}


def build_task(bench, part):
    if bench == "xstest":
        from inspect_evals.xstest import xstest
        return xstest(subset=part, scorer_model=JUDGE)
    if bench == "mask":
        from inspect_evals.mask import mask
        return mask(binary_judge_model=JUDGE, numeric_judge_model=JUDGE, judge_temperature=0.0)
    raise ValueError(bench)


def log_root():
    return Path(os.environ.get("EXTRA_LOG_ROOT", Path(os.environ["PROJ_DIR"]) / "logs_extra"))


def gen(args):
    think_max = int(os.environ["MAX_TOKENS_THINK"])
    nothink_max = int(os.environ["MAX_TOKENS_NOTHINK"])
    gen_cfg = dict(seed=args.seed, attempt_timeout=1800, timeout=3600,
                   extra_body={"chat_template_kwargs": {"enable_thinking": bool(args.think)}})
    if args.bench == "mask":
        # Keep original defaults (temperature 1.0, max_tokens 1000). Raise the limit only for think=1.
        if args.think:
            gen_cfg.update(max_tokens=think_max)
    else:
        gen_cfg.update(max_tokens=think_max if args.think else nothink_max)
        gen_cfg.update(temperature=0.0) if args.seed == 0 else gen_cfg.update(SAMPLING[args.think])

    policy = get_model(f"openai-api/policy/{args.tag}")
    for part in PARTS[args.bench]:
        log_dir = log_root() / args.bench / args.tag / f"think{args.think}" / f"seed{args.seed}" / part
        done = False
        for p in log_dir.glob("*.eval"):
            if read_eval_log(str(p), header_only=True).status == "success":
                done = True
            else:
                print(f"[clean] removing incomplete log: {p}")
                p.unlink(missing_ok=True)
        if done:
            print(f"[skip] {log_dir} already done")
            continue
        log_dir.mkdir(parents=True, exist_ok=True)
        print(f"[run] {args.bench} {args.tag} think={args.think} seed={args.seed} part={part}", flush=True)
        inspect_eval(
            build_task(args.bench, part), model=policy, log_dir=str(log_dir), log_format="eval",
            limit=args.limit, max_connections=int(os.environ.get("MAX_CONNECTIONS", "32")),
            fail_on_error=0.05, score=False,
            metadata=dict(bench=args.bench, tag=args.tag, think=args.think, seed=args.seed, part=part),
            **gen_cfg,
        )


def _restore_mask_enums(log):
    # After a round-trip through the log, enums become strings → fails the MASK scorer's strict pydantic validation.
    # Restore them to the same types as at generation time.
    # Model responses in the store (list[ModelOutput]) are also read back as dicts, so restore those too.
    from inspect_ai.model import ModelOutput
    from inspect_evals.mask.types import MetadataKey, QuestionArchetype, QuestionType, StoreKey
    keys = [v for k, v in vars(StoreKey).items() if not k.startswith("_")]
    for smp in log.samples:
        md = smp.metadata
        md[MetadataKey.CONFIG] = QuestionArchetype(md[MetadataKey.CONFIG])
        md[MetadataKey.TYPE] = QuestionType(md[MetadataKey.TYPE])
        for k in keys:
            v = smp.store.get(k)
            if isinstance(v, list):
                smp.store[k] = [ModelOutput.model_validate(x) if isinstance(x, dict) else x for x in v]


def score_all(args):
    root = log_root() / args.bench / args.tag
    for p in sorted(root.resolve().rglob("*.eval")):
        log = read_eval_log(str(p))
        if log.status != "success" or not log.samples:
            print(f"[warn] {p} status={log.status} — skipping"); continue
        if all(s.scores for s in log.samples):
            print(f"[skip] {p} already scored"); continue
        part = (log.eval.metadata or {}).get("part")
        if args.bench == "mask":
            _restore_mask_enums(log)
        scored = score(log, scorers=build_task(args.bench, part).scorer,
                       action="overwrite", display="plain")
        write_eval_log(scored, str(p))
        print(f"[ok] scored {p}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["gen", "score"])
    ap.add_argument("--bench", choices=list(PARTS), required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--think", type=int, choices=[0, 1])
    ap.add_argument("--seed", type=int)
    ap.add_argument("--limit", type=int, default=None)
    args = ap.parse_args()
    if args.mode == "gen":
        gen(args)
    else:
        score_all(args)


if __name__ == "__main__":
    main()
