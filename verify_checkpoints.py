"""
W1 (required): determine from the weights what each checkpoint was initialized from.

Three checks
  1) Architecture: is config.json really Qwen3-8B-shaped? (checks the suspicion that MARSHAL is 4B)
  2) Chat template: whether tools / <tool_call> / enable_thinking are supported
     -> without tool support in the template, tool calls are 'impossible' in AgentHarm, creating an illusion of safety
  3) Init check: compare layer-0 weights against both Qwen3-8B (instruct) and Qwen3-8B-Base
     RL updates are small, so a checkpoint is much closer to its own starting point.
     rel_dist = ||W_ckpt - W_ref|| / ||W_ref||  -> the smaller one is the starting point.

Downloaded shards stay in the HF cache and are reused by later evals (not wasted).
Run:  python verify_checkpoints.py            (all of models.tsv)
      python verify_checkpoints.py --tags marshal-gen   (subset of models)
      python verify_checkpoints.py --skip-weights   (config/template only, fast)
"""
import argparse
import json
import sys
from pathlib import Path

import torch
from huggingface_hub import hf_hub_download
from safetensors import safe_open

REF_INSTRUCT = "Qwen/Qwen3-8B"
REF_BASE = "Qwen/Qwen3-8B-Base"

# Expected Qwen3-8B values
EXPECTED = {"num_hidden_layers": 36, "hidden_size": 4096, "intermediate_size": 12288}

# Tensors to compare: all in layer 0 -> usually all in the first shard
PROBE_TENSORS = [
    "model.layers.0.self_attn.q_proj.weight",
    "model.layers.0.self_attn.o_proj.weight",
    "model.layers.0.mlp.down_proj.weight",
]


def load_models_tsv(path):
    rows = []
    for line in Path(path).read_text().splitlines():
        if not line.strip() or line.startswith("#"):
            continue
        tag, repo, group, it = line.split("\t")
        rows.append({"tag": tag, "repo": repo, "group": group, "iter": int(it)})
    return rows


def read_json(repo, fname):
    try:
        return json.loads(Path(hf_hub_download(repo, fname)).read_text())
    except Exception as e:
        return {"__error__": str(e)}


def check_config(repo):
    cfg = read_json(repo, "config.json")
    if "__error__" in cfg:
        return {"config_error": cfg["__error__"]}
    out = {k: cfg.get(k) for k in EXPECTED}
    out["arch"] = (cfg.get("architectures") or ["?"])[0]
    out["name_or_path"] = cfg.get("_name_or_path", "")
    out["arch_ok"] = all(cfg.get(k) == v for k, v in EXPECTED.items())
    return out


def check_template(repo):
    tok = read_json(repo, "tokenizer_config.json")
    tmpl = tok.get("chat_template", "") if "__error__" not in tok else ""
    if isinstance(tmpl, list):  # some models use a list of named templates
        tmpl = " ".join(t.get("template", "") for t in tmpl)
    if not tmpl:
        # recent models ship a separate chat_template.jinja file
        try:
            tmpl = Path(hf_hub_download(repo, "chat_template.jinja")).read_text()
        except Exception:
            tmpl = ""
    return {
        "has_template": bool(tmpl),
        "tmpl_tools": "tools" in tmpl,
        "tmpl_tool_call_tag": "<tool_call>" in tmpl,
        "tmpl_enable_thinking": "enable_thinking" in tmpl,
    }


def load_probe_tensors(repo):
    idx = read_json(repo, "model.safetensors.index.json")
    if "__error__" in idx:
        files = {name: "model.safetensors" for name in PROBE_TENSORS}
    else:
        files = {name: idx["weight_map"].get(name) for name in PROBE_TENSORS}
    out = {}
    for name, fname in files.items():
        if fname is None:
            continue
        local = hf_hub_download(repo, fname)
        with safe_open(local, framework="pt") as f:
            if name in f.keys():
                out[name] = f.get_tensor(name).float()
    return out


def rel_dist(a, b):
    num = sum(torch.linalg.norm(a[k] - b[k]).item() ** 2 for k in a if k in b) ** 0.5
    den = sum(torch.linalg.norm(b[k]).item() ** 2 for k in a if k in b) ** 0.5
    return num / den if den else float("nan")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--models", default=str(Path(__file__).parent / "models.tsv"))
    ap.add_argument("--skip-weights", action="store_true")
    ap.add_argument("--tags", nargs="+", help="check only some tags from models.tsv (e.g. --tags marshal-gen)")
    ap.add_argument("--out", default="results/checkpoint_verification.json")
    args = ap.parse_args()

    rows = load_models_tsv(args.models)
    if args.tags:
        rows = [r for r in rows if r["tag"] in args.tags]
        if not rows:
            sys.exit(f"no rows match --tags {args.tags}")
    report = []

    refs = {}
    if not args.skip_weights:
        print("loading reference tensors ...", flush=True)
        refs["instruct"] = load_probe_tensors(REF_INSTRUCT)
        refs["base"] = load_probe_tensors(REF_BASE)
        d_ref = rel_dist(refs["instruct"], refs["base"])
        print(f"[sanity] rel_dist(Qwen3-8B, Qwen3-8B-Base) = {d_ref:.4f}  "
              f"(this must be much larger than the checkpoint distances for the verdict to be meaningful)\n")

    for r in rows:
        rec = {**r, **check_config(r["repo"]), **check_template(r["repo"])}
        if not args.skip_weights and r["group"] != "ref" and rec.get("arch_ok"):
            w = load_probe_tensors(r["repo"])
            rec["dist_to_instruct"] = rel_dist(w, refs["instruct"])
            rec["dist_to_base"] = rel_dist(w, refs["base"])
            di, db = rec["dist_to_instruct"], rec["dist_to_base"]
            # Require a 3x or larger gap to decide; otherwise a human has to look.
            if di * 3 < db:
                rec["init_verdict"] = "Qwen3-8B (instruct, aligned)"
            elif db * 3 < di:
                rec["init_verdict"] = "Qwen3-8B-Base (pre-alignment!)"
            else:
                rec["init_verdict"] = "AMBIGUOUS — confirm with authors"
        report.append(rec)

        flags = []
        if not rec.get("arch_ok"):
            flags.append("❌ not an 8B architecture")
        if not rec.get("tmpl_tools"):
            flags.append("❌ no tool support in template")
        if not rec.get("tmpl_enable_thinking"):
            flags.append("⚠️ no enable_thinking toggle (think on/off comparison impossible)")
        print(f"{r['tag']:20s} layers={rec.get('num_hidden_layers')} hidden={rec.get('hidden_size')} "
              f"init={rec.get('init_verdict', '-')} "
              f"d_inst={rec.get('dist_to_instruct', float('nan')):.4f} "
              f"d_base={rec.get('dist_to_base', float('nan')):.4f}  {' '.join(flags)}", flush=True)

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2, ensure_ascii=False))
    print(f"\nsaved -> {args.out}")

    bad = [x["tag"] for x in report if not x.get("arch_ok")]
    if bad:
        print(f"\n🚨 architecture mismatch: {bad} — exclude these models from comparisons or handle separately")
        sys.exit(2)


if __name__ == "__main__":
    main()
