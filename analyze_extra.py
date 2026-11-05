"""
XSTest / MASK results summary (run_extra.py logs → results/extra_*.csv)

XSTest
  refusal      = fraction judged I (full refusal) or P (partial refusal)  (original paper definition)
  full_refusal = fraction judged I
  refusal on the safe subset = over-refusal (lower is better), refusal on the unsafe subset = correct refusal (higher is better)
  The log's refusal_rate metric is not used: the rescoring path doesn't convert letters to numbers, so it comes out as 0.
MASK
  lie        = fraction stating something contrary to its own belief under pressure (MASK overall_honesty = 1 - lie)
  honest / evade / no_belief = fractions of the remaining categories
  accuracy   = overall_accuracy from the log (fraction of beliefs matching ground truth, MASK definition)

Differences vs qwen3-8b are computed on seed 0 with a paired bootstrap (paired by sample id) 95% CI.
"""
import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd
from inspect_ai.log import read_eval_log

ROOT = Path(os.environ.get("EXTRA_LOG_ROOT", "logs_extra"))
OUT = Path(os.environ.get("RESULTS_DIR", "results"))
REF = "qwen3-8b"
N_BOOT = 2000
rng = np.random.default_rng(0)


def load():
    rows = []
    for p in sorted(glob.glob(str(ROOT / "*/*/think*/seed*/*/*.eval"))):
        bench, tag, think, seed, part = Path(p).parts[-6:-1]
        log = read_eval_log(p)
        if log.status != "success":
            continue
        acc = None
        if bench == "mask" and log.results:
            acc = log.results.scores[0].metrics["overall_accuracy"].value
        for s in log.samples:
            v = list(s.scores.values())[0].value
            r = dict(bench=bench, tag=tag, think=int(think[5:]), seed=int(seed[4:]), part=part, id=str(s.id))
            if bench == "xstest":
                r.update(refusal=float(v in ("I", "P")), full_refusal=float(v == "I"))
            else:
                h = v["honesty"]
                r.update(lie=float(h == "lie"), honest=float(h == "honest"), evade=float(h == "evade"),
                         no_belief=float(h == "no-belief"), accuracy_log=acc)
            rows.append(r)
    return pd.DataFrame(rows)


def boot_ci(x):
    idx = rng.integers(0, len(x), (N_BOOT, len(x)))
    m = x[idx].mean(1)
    return np.percentile(m, [2.5, 97.5])


def main():
    df = load()
    df.to_csv(OUT / "extra_samples.csv", index=False)
    summ, delta = [], []
    for bench, metrics in [("xstest", ["refusal", "full_refusal"]), ("mask", ["lie", "honest", "evade", "no_belief"])]:
        d = df[df.bench == bench]
        for (tag, think, part), g in d.groupby(["tag", "think", "part"]):
            g0 = g[g.seed == 0]
            r = dict(bench=bench, tag=tag, think=think, part=part, n=len(g0))
            for m in metrics:
                r[m] = g0[m].mean()
                r[m + "_lo"], r[m + "_hi"] = boot_ci(g0[m].to_numpy())
                per_seed = g[g.seed > 0].groupby("seed")[m].mean()
                if len(per_seed):
                    r[m + "_sampled_mean"], r[m + "_sampled_std"] = per_seed.mean(), per_seed.std()
            if bench == "mask":
                r["accuracy"] = g0["accuracy_log"].iloc[0]
            summ.append(r)
            if tag == REF:
                continue
            ref = d[(d.tag == REF) & (d.think == think) & (d.part == part) & (d.seed == 0)]
            pair = g0.merge(ref, on="id", suffixes=("", "_ref"))
            for m in metrics[:1] if bench == "xstest" else ["lie", "honest"]:
                diff = (pair[m] - pair[m + "_ref"]).to_numpy()
                lo, hi = boot_ci(diff)
                delta.append(dict(bench=bench, tag=tag, think=think, part=part, metric=m, ref=REF,
                                  delta=diff.mean(), lo=lo, hi=hi, n_pairs=len(pair),
                                  significant=bool(lo > 0 or hi < 0)))
    pd.DataFrame(summ).to_csv(OUT / "extra_summary.csv", index=False)
    pd.DataFrame(delta).to_csv(OUT / "extra_delta.csv", index=False)
    print(f"saved -> {OUT}/extra_summary.csv, extra_delta.csv, extra_samples.csv  ({len(df)} rows)")


if __name__ == "__main__":
    main()
