"""
Judge validation (for reviewers): agreement between human labels and the Gemma-3-27B judge.

1) Draw samples (stratified by model/think/split, 150 by default)
   python judge_validation.py sample --n 150
   -> creates results/human_labels_agentharm.csv. Fill in the human_refusal column with 0/1 by hand.
      Judge verdicts are kept separately in results/judge_key_agentharm.csv (do not open until labeling is done).
2) Compute agreement
   python judge_validation.py kappa
   -> Cohen's kappa, confusion matrix. If kappa >= 0.8 report as-is in the paper; if below 0.6 consider replacing the judge.
"""
import argparse
import os
from pathlib import Path

import pandas as pd
from inspect_ai.log import read_eval_log

RES = Path(os.environ.get("RESULTS_DIR", "results"))
LOG_ROOT = Path(os.environ.get("LOG_ROOT", "logs")).resolve()


def visible(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return "\n".join(getattr(x, "text", "") or "" for x in c if getattr(x, "type", "") == "text")


def cmd_sample(n):
    df = pd.read_csv(RES / "samples.csv")
    df = df[(df.seed == 0) & df.refusal_visible.notna()]
    per = max(1, n // df.groupby(["tag", "think", "split"]).ngroups)
    pick = df.groupby(["tag", "think", "split"], group_keys=False).sample(n=per, random_state=0)

    rows = []
    for (tag, think, split), g in pick.groupby(["tag", "think", "split"]):
        log_path = next((LOG_ROOT / tag / f"think{think}" / "seed0" / split).glob("*.eval"))
        log = read_eval_log(str(log_path))
        by_id = {str(s.id): s for s in log.samples}
        for _, r in g.iterrows():
            s = by_id[str(r.id)]
            asst = [m for m in s.messages if m.role == "assistant"]
            calls = "; ".join(f"{tc.function}" for m in asst for tc in (m.tool_calls or []))
            rows.append(dict(
                tag=tag, think=think, split=split, id=r.id,
                prompt=s.input if isinstance(s.input, str) else visible(s.input[-1]),
                visible_response="\n---\n".join(visible(m) for m in asst if visible(m).strip())[:4000],
                tool_calls=calls, judge_refusal=int(r.refusal_visible), human_refusal="",
            ))
    allr = pd.DataFrame(rows).sample(frac=1, random_state=1).reset_index(drop=True)  # shuffle order
    allr.insert(0, "row", range(len(allr)))
    # Avoid anchoring/model bias: the label sheet omits judge verdicts and model/think; those go in a separate key file
    allr[["row", "tag", "think", "split", "id", "judge_refusal"]].to_csv(RES / "judge_key_agentharm.csv", index=False)
    out = RES / "human_labels_agentharm.csv"
    allr.drop(columns=["tag", "think", "judge_refusal"]).to_csv(out, index=False)
    print(f"saved {len(rows)} rows -> {out}  (key: judge_key_agentharm.csv, do not open until labeling is done)")


def cmd_kappa():
    df = (pd.read_csv(RES / f"{os.environ.get('LABELS_PREFIX', 'human_labels')}_agentharm.csv")
          .merge(pd.read_csv(RES / "judge_key_agentharm.csv"), on="row").dropna(subset=["human_refusal"]))
    h, j = df.human_refusal.astype(int), df.judge_refusal.astype(int)
    po = (h == j).mean()
    pe = h.mean() * j.mean() + (1 - h.mean()) * (1 - j.mean())
    kappa = (po - pe) / (1 - pe) if pe < 1 else float("nan")
    print(f"n={len(df)}  agreement={po:.3f}  Cohen's kappa={kappa:.3f}")
    print(pd.crosstab(h, j, rownames=["human"], colnames=["judge"]))
    print("\nCheck disagreeing samples in human_labels.csv → summarize error types in the paper appendix")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("sample")
    s.add_argument("--n", type=int, default=150)
    sub.add_parser("kappa")
    a = ap.parse_args()
    cmd_sample(a.n) if a.cmd == "sample" else cmd_kappa()
