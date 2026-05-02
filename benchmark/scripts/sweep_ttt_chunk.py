"""Sweep ttt_chunk at eval time and re-run a task across context lengths.

Used to test the hypothesis that the LongAlpacaTTT bump on CWE at 8k/16k is
governed by the number of TTT chunks (= ⌈seq_len / ttt_chunk⌉) rather than by
seq_len itself. If the bump tracks chunk count, it should *move* with ttt_chunk.

Loads the checkpoint once per ttt_chunk value (config kwargs to from_pretrained
override the saved field), then evaluates across the chosen context lengths.

Usage:
    python -m benchmark.scripts.sweep_ttt_chunk \\
        --repo changminbark/gemma-3-1b-it-ttt-longalpaca-full \\
        --task cwe \\
        --lengths 4096,8192,16384,32768 \\
        --ttt-chunks 1024,2048,4096,8192,16384 \\
        --limit 100
"""

from __future__ import annotations

import argparse
import csv
import os
from pathlib import Path

import torch
import yaml

from ..eval.gemma3_predictors import (
    _get_hf_token,
    build_generate_subclass,
    make_generate_fn,
)
from ..eval.predictor import SinglePassPredictor
from ..eval.runner import run_benchmark


def load_with_chunk(repo: str, ttt_chunk: int):
    """Same as gemma3_predictors.load_gemma3_ttt_model but with ttt_chunk override.

    Workaround: transformers' loader silently drops the ttt_conv/ttt_proj
    checkpoint tensors as UNEXPECTED, even though the model *does* register
    those parameters. We manually re-load them via load_state_dict(strict=False)
    after from_pretrained.
    """
    from transformers import AutoTokenizer
    from huggingface_hub import hf_hub_download
    from safetensors import safe_open
    from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig

    cls = build_generate_subclass()
    token = _get_hf_token()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    torch_dtype = torch.bfloat16 if device == "cuda" else torch.float32

    cfg = Gemma3TTTConfig.from_pretrained(
        repo,
        token=token,
        use_ttt=True,
        ttt_layers=[0, 6, 12, 18, 24],
        ttt_chunk=ttt_chunk,
    )
    assert cfg.ttt_chunk == ttt_chunk, f"override failed: {cfg.ttt_chunk}"

    model = cls.from_pretrained(
        repo, config=cfg, token=token, torch_dtype=torch_dtype
    ).to(device).eval()

    # Manually load TTT tensors that the standard loader dropped. Read directly
    # from the safetensors shard(s) and copy any "ttt_*" key into the model.
    ckpt_path = hf_hub_download(repo, "model.safetensors", token=token)
    ttt_state: dict[str, torch.Tensor] = {}
    with safe_open(ckpt_path, framework="pt") as f:
        for k in f.keys():
            if "ttt_proj" in k or "ttt_conv" in k:
                ttt_state[k] = f.get_tensor(k).to(device=device, dtype=torch_dtype)
    if ttt_state:
        result = model.load_state_dict(ttt_state, strict=False)
        # missing_keys here = all non-TTT keys in the model; that's expected
        # because we only passed TTT tensors. unexpected_keys should be empty.
        if result.unexpected_keys:
            print(f"[load_with_chunk] still-unexpected after manual load: "
                  f"{result.unexpected_keys[:5]}{'...' if len(result.unexpected_keys) > 5 else ''}")

    tok = AutoTokenizer.from_pretrained(repo, token=token)
    return model, tok


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--repo", default="changminbark/gemma-3-1b-it-ttt-longalpaca-full")
    p.add_argument("--config", default="benchmark/configs/benchmark.yaml")
    p.add_argument("--task", default="cwe")
    p.add_argument("--lengths", default="4096,8192,16384,32768")
    p.add_argument("--ttt-chunks", default="1024,2048,4096,8192,16384")
    p.add_argument("--data-root", default=None,
                   help="dir holding <task>_<L>.jsonl. If unset, tries dev/full/(root).")
    p.add_argument("--results-root", default="benchmark/results/raw/chunk_sweep")
    p.add_argument("--limit", type=int, default=None)
    p.add_argument("--diagnose", action="store_true",
                   help="Print TTT module weight norms and an in_context (use_ttt=False) "
                        "baseline run on the same checkpoint, then exit.")
    args = p.parse_args()

    # Resolve cfg + data root first; --diagnose uses both.
    cfg = yaml.safe_load(Path(args.config).read_text())
    lengths_pre = [int(x) for x in args.lengths.split(",") if x.strip()]
    if args.data_root is None:
        candidates = ["benchmark/data/dev", "benchmark/data/full", "benchmark/data"]
        for cand in candidates:
            if any((Path(cand) / f"{args.task}_{L}.jsonl").exists() for L in lengths_pre):
                args.data_root = cand
                break
        if args.data_root is None:
            print(f"[fatal] no {args.task}_<L>.jsonl found under any of: {candidates}")
            print("        Pass --data-root <dir> explicitly. Existing benchmark/data tree:")
            os.system("find benchmark/data -maxdepth 3 -name '*.jsonl' 2>/dev/null | head -30")
            return
        print(f"[info] using --data-root {args.data_root}")

    if args.diagnose:
        # Inspect checkpoint state_dict keys before loading the model. The load
        # report flagged ttt_proj/ttt_conv as UNEXPECTED — meaning the keys in
        # the checkpoint don't match what the model expects. Print both sides.
        from huggingface_hub import hf_hub_download
        from safetensors import safe_open
        import json as _json
        token = _get_hf_token()

        # 1. Saved config.json — tells us what arch the loader thinks it's building.
        try:
            cfg_path = hf_hub_download(args.repo, "config.json", token=token)
            saved_cfg = _json.loads(Path(cfg_path).read_text())
            print("\n--- Saved config.json TTT fields ---")
            for k in ("use_ttt", "ttt_layers", "ttt_chunk", "ttt_lr", "ttt_proj", "ttt_target", "architectures", "model_type"):
                print(f"  {k}: {saved_cfg.get(k, '<absent>')}")
        except Exception as e:
            print(f"[diagnose] could not read saved config.json: {e}")

        # 2. Checkpoint TTT keys + values' L2 (proves the ckpt has real weights).
        try:
            ckpt_path = hf_hub_download(args.repo, "model.safetensors", token=token)
            with safe_open(ckpt_path, framework="pt") as f:
                ckpt_keys = list(f.keys())
                ttt_keys_in_ckpt = sorted(k for k in ckpt_keys if "ttt" in k.lower())
                print(f"\n--- Checkpoint TTT-related keys ({len(ttt_keys_in_ckpt)}) ---")
                for k in ttt_keys_in_ckpt[:20]:
                    t = f.get_tensor(k)
                    print(f"  CKPT: {k}  shape={tuple(t.shape)}  L2={t.float().norm().item():.4e}")
                if len(ttt_keys_in_ckpt) > 20:
                    print(f"  ... ({len(ttt_keys_in_ckpt) - 20} more)")
        except Exception as e:
            print(f"[diagnose] could not inspect checkpoint: {e}")

        # Load once with TTT on at the saved chunk size, dump norms.
        m, tok = load_with_chunk(args.repo, ttt_chunk=2048)
        model_ttt_keys = sorted(
            n + ".weight" for n, _ in m.named_modules()
            if n.endswith(("ttt_proj", "ttt_conv"))
        )
        print(f"\n--- Model expected TTT keys ({len(model_ttt_keys)}) ---")
        for k in model_ttt_keys[:20]:
            print(f"  MODEL: {k}")

        # Confirm whether the post-load model state_dict actually contains the
        # TTT keys (it should, since named_modules sees them; if not, the load
        # report's UNEXPECTED warning is hiding a deeper issue).
        sd_keys = set(m.state_dict().keys())
        in_sd = [k for k in model_ttt_keys if k in sd_keys]
        print(f"\n--- TTT keys present in model.state_dict() post-load: {len(in_sd)}/{len(model_ttt_keys)} ---")
        print("\n--- TTT module weight stats (use_ttt=True) ---")
        for name, mod in m.named_modules():
            if name.endswith(("ttt_conv", "ttt_proj")):
                w = mod.weight
                print(f"  {name:60s}  shape={tuple(w.shape)}  "
                      f"L2={w.float().norm().item():.4e}  "
                      f"max|w|={w.float().abs().max().item():.4e}")
        # Also: the actual W_down used at chunk 0 is just down_proj.weight; if it
        # diverges from base Gemma, that explains accuracy gap independent of TTT.
        print("\n--- down_proj.weight L2 (first 3 ttt-layers) ---")
        for name, mod in m.named_modules():
            if name.endswith("mlp.down_proj") and "layers." in name:
                idx = int(name.split("layers.")[1].split(".")[0])
                if idx in (0, 6, 12):
                    w = mod.weight
                    print(f"  {name}  L2={w.float().norm().item():.4e}")
        del m, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        # Now run with use_ttt=False on the SAME checkpoint at one L. If acc
        # equals the ttt_paper acc above, the chunked update path contributes
        # nothing and the longalpaca improvement is purely from drifted weights.
        from ..eval.gemma3_predictors import load_gemma3_ttt_model
        m2, tok2 = load_gemma3_ttt_model(args.repo, use_ttt=False)
        pred_no_ttt = SinglePassPredictor(
            model_name=f"{args.repo.split('/')[-1]}-NO_TTT",
            mode="in_context",
            generate_fn=make_generate_fn(m2, tok2),
        )
        L = 8192
        data_path = Path(args.data_root) / f"cwe_{L}.jsonl"
        if not data_path.exists():
            print(f"[diagnose] missing {data_path}; pass --data-root if elsewhere")
            return
        results_path = Path("benchmark/results/raw/chunk_sweep") / "DIAGNOSE_no_ttt" / f"cwe_{L}.jsonl"
        scoring_opts = {
            "lowercase": cfg["scoring"]["normalize_case"],
            "strip_whitespace": cfg["scoring"]["strip_whitespace"],
            "strip_punctuation": cfg["scoring"]["strip_punctuation"],
        }
        s = run_benchmark(
            dataset_path=data_path, results_path=results_path, predictor=pred_no_ttt,
            max_new_tokens=120, limit=args.limit, scoring_opts=scoring_opts,
        )
        print(f"\n--- {pred_no_ttt.model_name} @ L={L} (use_ttt=False on same ckpt) ---")
        print(f"  n={s['n']}  acc={s['accuracy']:.3f}")
        print("  Compare to ttt_paper @ L=8192 = 0.640. Equal -> TTT path is no-op.")
        return

    default_max_new = cfg["generation"]["max_new_tokens"]
    per_task_max_new = (cfg.get("generation") or {}).get("max_new_tokens_per_task") or {}
    max_new = int(per_task_max_new.get(args.task, default_max_new))
    scoring_opts = {
        "lowercase": cfg["scoring"]["normalize_case"],
        "strip_whitespace": cfg["scoring"]["strip_whitespace"],
        "strip_punctuation": cfg["scoring"]["strip_punctuation"],
    }

    lengths = [int(x) for x in args.lengths.split(",") if x.strip()]
    chunks = [int(x) for x in args.ttt_chunks.split(",") if x.strip()]
    short = args.repo.rstrip("/").split("/")[-1]

    table: dict[tuple[int, int], float] = {}
    summary_path = Path("benchmark/results/summary") / f"chunk_sweep__{short}__{args.task}.csv"
    summary_path.parent.mkdir(parents=True, exist_ok=True)

    for chunk in chunks:
        print(f"\n=== ttt_chunk={chunk} ===")
        model, tok = load_with_chunk(args.repo, chunk)
        predictor = SinglePassPredictor(
            model_name=f"{short}-ttt-chunk{chunk}",
            mode="ttt_paper",
            generate_fn=make_generate_fn(model, tok),
        )
        for L in lengths:
            data_path = Path(args.data_root) / f"{args.task}_{L}.jsonl"
            if not data_path.exists():
                print(f"[skip] missing {data_path}  (try a different --data-root)")
                continue
            results_path = (
                Path(args.results_root)
                / f"{predictor.model_name}__{predictor.mode}"
                / f"{args.task}_{L}.jsonl"
            )
            n_chunks = -(-L // chunk)  # ceil
            summary = run_benchmark(
                dataset_path=data_path,
                results_path=results_path,
                predictor=predictor,
                max_new_tokens=max_new,
                limit=args.limit,
                scoring_opts=scoring_opts,
            )
            table[(chunk, L)] = summary["accuracy"]
            print(
                f"  L={L:>6}  ~chunks={n_chunks:>3}  "
                f"n={summary['n']:>3}  acc={summary['accuracy']:.3f}  "
                f"latency_ms={summary['mean_latency_ms']:.0f}"
            )
        del model, tok
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    with summary_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["ttt_chunk"] + [f"L={L}" for L in lengths])
        for chunk in chunks:
            w.writerow([chunk] + [f"{table.get((chunk, L), float('nan')):.3f}" for L in lengths])
    print(f"\n[done] wrote {summary_path}")


if __name__ == "__main__":
    main()
