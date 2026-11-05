"""
Read all logs and build the paper's tables and figures. (No GPU needed; runs on a login node)

  python analyze.py                    # reference = qwen3-8b
  python analyze.py --ref qwen3-8b-base  # if SPIRAL turns out to start from Base

Outputs (results/)
  samples.csv        per-sample raw data (starting point for all analyses)
  summary.csv        metrics per (tag, think, split) + 95% CI
  delta_vs_ref.csv   paired difference vs reference model + CI  ← main-text table
  spiral_curve.png   dose-response over SPIRAL iters  ← Figure 1 candidate

Metric definitions (the 4 must be reported separately, to distinguish 'compliance' from 'capability')
  refusal_visible   refusal rate on visible text (primary metric, rescore_visible.py)
  refusal_official  original benchmark refusal rate (judged with reasoning included, for comparison)
  harm_score        mean rubric score on the harmful split
  nonref_score      score on non-refused samples only = 'execution capability'
  benign_score      benign split score = pure capability control
  parse_fail_rate   tool call written as text but failed to parse → watches for false 'safe' readings
  trunc_rate        truncated at max_tokens → also watches for false 'no action' readings

CI: the 4 variants (hint/detailed combinations) of the same base behavior are not independent,
    so use a behavior-level cluster bootstrap.
"""
import argparse
import json
import os
import re
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
from inspect_ai.log import read_eval_log  # noqa: E402

RNG = np.random.default_rng(0)
N_BOOT = 2000
TOOLCALL_TEXT = re.compile(r"<tool_call>|\"arguments\"\s*:", re.I)


def load_models(path):
    rows = [l.split("\t") for l in Path(path).read_text().splitlines()
            if l.strip() and not l.startswith("#")]
    return pd.DataFrame(rows, columns=["tag", "repo", "group", "iter"]).astype({"iter": int})


def text_of(msg):
    c = msg.content
    if isinstance(c, str):
        return c
    return "\n".join(getattr(x, "text", "") or "" for x in c if getattr(x, "type", "") == "text")


def read_samples(log_root: Path, vis_root: Path) -> pd.DataFrame:
    recs = []
    for p in sorted(log_root.rglob("*.eval")):
        log = read_eval_log(str(p))
        if log.status != "success" or not log.samples:
            print(f"[skip] {p} status={log.status}")
            continue
        meta = log.eval.metadata or {}
        vis_file = vis_root / p.relative_to(log_root).with_suffix(".visible.json")
        vis = {}
        if vis_file.exists():
            vis = {(r["id"], r["epoch"]): r["refusal_visible"] for r in json.loads(vis_file.read_text())}
        for s in log.samples:
            score = next(iter(s.scores.values())).value if s.scores else {}
            asst = [m for m in s.messages if m.role == "assistant"]
            n_calls = sum(len(m.tool_calls or []) for m in asst)
            parse_fail = any(TOOLCALL_TEXT.search(text_of(m)) and not m.tool_calls for m in asst)
            stop = getattr(s.output, "stop_reason", None) if s.output else None
            sid = str(s.id)
            recs.append(dict(
                tag=meta.get("tag"), think=meta.get("think"), seed=meta.get("seed"), split=meta.get("split"),
                id=sid, behavior=sid.split("-")[0], category=(s.metadata or {}).get("category"),
                score=score.get("score", np.nan),
                refusal_official=score.get("refusal", np.nan),
                refusal_visible=vis.get((sid, s.epoch), np.nan),
                n_tool_calls=n_calls, parse_fail=int(parse_fail),
                truncated=int(stop == "max_tokens"),
            ))
    return pd.DataFrame(recs)


def cluster_boot(df, col, clusters="behavior"):
    """95% CI of the mean, resampling at the behavior level."""
    g = df.groupby(clusters)[col].agg(["sum", "count"])
    if g.empty or g["count"].sum() == 0:
        return np.nan, np.nan
    idx = RNG.integers(0, len(g), size=(N_BOOT, len(g)))
    s, c = g["sum"].to_numpy()[idx].sum(1), g["count"].to_numpy()[idx].sum(1)
    est = s / np.maximum(c, 1)
    return np.nanpercentile(est, 2.5), np.nanpercentile(est, 97.5)


def summarize(df):
    out = []
    for (tag, think, split), g in df[df.seed == 0].groupby(["tag", "think", "split"]):
        row = dict(tag=tag, think=think, split=split, n=len(g))
        for col in ["refusal_visible", "refusal_official", "score", "parse_fail", "truncated"]:
            sub = g.dropna(subset=[col])
            row[col] = sub[col].mean() if len(sub) else np.nan
            row[f"{col}_lo"], row[f"{col}_hi"] = cluster_boot(sub, col) if len(sub) else (np.nan, np.nan)
        nr = g[g.refusal_visible == 0].dropna(subset=["score"])
        row["nonref_score"] = nr.score.mean() if len(nr) else np.nan
        # variation across sampling seeds (seed>0)
        sampled = df[(df.tag == tag) & (df.think == think) & (df.split == split) & (df.seed > 0)]
        per_seed = sampled.groupby("seed").refusal_visible.mean()
        row["refusal_visible_sampled_mean"] = per_seed.mean() if len(per_seed) else np.nan
        row["refusal_visible_sampled_std"] = per_seed.std() if len(per_seed) > 1 else np.nan
        out.append(row)
    return pd.DataFrame(out)


def paired_delta(df, ref):
    """(model - reference) difference paired by sample id, with behavior cluster bootstrap CI."""
    out = []
    base = df[(df.tag == ref) & (df.seed == 0)]
    for (tag, think, split), g in df[(df.seed == 0) & (df.tag != ref)].groupby(["tag", "think", "split"]):
        b = base[(base.think == think) & (base.split == split)]
        for col in ["refusal_visible", "score"]:
            m = g[["id", "behavior", col]].merge(b[["id", col]], on="id", suffixes=("", "_ref")).dropna()
            if m.empty:
                continue
            m["d"] = m[col] - m[f"{col}_ref"]
            lo, hi = cluster_boot(m, "d")
            out.append(dict(tag=tag, think=think, split=split, metric=col, ref=ref,
                            delta=m.d.mean(), lo=lo, hi=hi, n_pairs=len(m),
                            significant=bool(lo > 0 or hi < 0)))
    return pd.DataFrame(out)


def plot_curve(summ, models, ref, out_png):
    """Small multiples: each metric gets its own axis (no dual-axis). think off/on = 2 series."""
    sp = models[models.group == "spiral"][["tag", "iter"]]
    ref_row = pd.DataFrame([{"tag": ref, "iter": 0}])
    xs = pd.concat([ref_row, sp])
    panels = [("harmful", "refusal_visible", "Refusal rate (harmful)"),
              ("harmful", "score", "Harm score (harmful)"),
              ("benign", "score", "Task score (benign)")]
    series = [(0, "Thinking off", "#2a78d6", "o"), (1, "Thinking on", "#eb6834", "s")]

    fig, axes = plt.subplots(1, 3, figsize=(13, 3.6), sharex=True)
    fig.patch.set_facecolor("#fcfcfb")
    for ax, (split, col, title) in zip(axes, panels):
        ax.set_facecolor("#fcfcfb")
        placed = []  # end-label y positions (pushed apart on overlap)
        for think, label, color, marker in series:
            d = summ[(summ.split == split) & (summ.think == think)].merge(xs, on="tag").sort_values("iter")
            if d.empty:
                continue
            ax.fill_between(d["iter"], d[f"{col}_lo"], d[f"{col}_hi"], color=color, alpha=0.15, lw=0)
            ax.plot(d["iter"], d[col], color=color, lw=2, marker=marker, ms=8,
                    markeredgecolor="#fcfcfb", markeredgewidth=2, label=label)
            y_end = d[col].iloc[-1]
            dy = 0
            while any(abs((y_end + dy / 100) - p) < 0.05 for p in placed):
                dy += 5
            placed.append(y_end + dy / 100)
            ax.annotate(label, (d["iter"].iloc[-1], y_end), xytext=(6, dy * 2),
                        textcoords="offset points", va="center", fontsize=8, color="#3d3d3a")
        ax.set_title(title, fontsize=10, color="#1f1f1e", loc="left")
        ax.set_ylim(0, 1)
        ax.set_xlabel("SPIRAL training iteration (0 = reference)", fontsize=8, color="#5e5d59")
        ax.grid(axis="y", color="#e8e6dc", lw=0.8)
        for s in ["top", "right"]:
            ax.spines[s].set_visible(False)
        for s in ["left", "bottom"]:
            ax.spines[s].set_color("#c3c2b7")
        ax.tick_params(colors="#5e5d59", labelsize=8)
    axes[0].legend(frameon=False, fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(out_png, dpi=200, facecolor=fig.get_facecolor())
    print(f"saved -> {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ref", default="qwen3-8b")
    args = ap.parse_args()

    log_root = Path(os.environ.get("LOG_ROOT", "logs")).resolve()
    res = Path(os.environ.get("RESULTS_DIR", "results"))
    res.mkdir(parents=True, exist_ok=True)
    models = load_models(Path(__file__).parent / "models.tsv")

    df = read_samples(log_root, res / "visible_refusal")
    if df.empty:
        raise SystemExit("No logs found. Run eval_array.sbatch first.")
    df.to_csv(res / "samples.csv", index=False)

    missing_vis = df.refusal_visible.isna().mean()
    if missing_vis > 0:
        print(f"[warn] visible rescoring missing for {missing_vis:.1%} — rerun rescore_visible.py")

    summ = summarize(df)
    summ.to_csv(res / "summary.csv", index=False)
    delta = paired_delta(df, args.ref)
    delta.to_csv(res / "delta_vs_ref.csv", index=False)

    # Illusion alarm: if parse failures/truncation exceed 5%, that model's 'safety' can't be trusted
    alarm = summ[(summ.parse_fail > 0.05) | (summ.truncated > 0.05)]
    if len(alarm):
        print("\n🚨 parse_fail or truncated > 5% — do not interpret refusal/harm for these rows:")
        print(alarm[["tag", "think", "split", "parse_fail", "truncated"]].to_string(index=False))

    with pd.option_context("display.width", 200, "display.max_columns", 20):
        print("\n", summ[["tag", "think", "split", "n", "refusal_visible", "refusal_official",
                          "score", "nonref_score", "parse_fail", "truncated"]].round(3).to_string(index=False))

    plot_curve(summ, models, args.ref, res / "spiral_curve.png")


if __name__ == "__main__":
    main()
