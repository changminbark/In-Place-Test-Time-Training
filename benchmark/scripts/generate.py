"""Generate benchmark JSONL files for each task × context length.

Usage:
    python -m benchmark.scripts.generate --profile dev
    python -m benchmark.scripts.generate --profile full
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import yaml

from ..data_gen import gen_multi_needle, gen_single_needle, gen_variable_tracking


_TOKEN_ENV_VARS = ("HF_TOKEN", "HUGGINGFACE_HUB_TOKEN", "HUGGING_FACE_HUB_TOKEN")


def _load_dotenv_if_present() -> None:
    import os
    from pathlib import Path as _P
    env_path = _P(__file__).resolve().parents[2] / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        k, _, v = line.partition("=")
        os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))


def _get_hf_token():
    import os
    _load_dotenv_if_present()
    for var in _TOKEN_ENV_VARS:
        tok = os.environ.get(var)
        if tok:
            return tok
    return None


def _load_tokenizer(model_id: str):
    try:
        from transformers import AutoTokenizer
    except ImportError as e:
        raise RuntimeError(
            "transformers is required for benchmark data generation. "
            "Install with `pip install transformers`."
        ) from e
    token = _get_hf_token()
    return AutoTokenizer.from_pretrained(model_id, token=token)


def _gen_for_task(task, tokenizer, target_tokens, n_examples, seed, positions):
    """Yield n_examples examples for a task at a given context length."""
    for i in range(n_examples):
        if task == "single_needle":
            pos = positions[i % len(positions)]
            yield gen_single_needle(
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                position=pos,
                seed=seed,
                example_idx=i,
            )
        elif task == "multi_needle":
            pos_1 = positions[i % len(positions)]
            pos_2 = positions[(i + 1) % len(positions)]
            yield gen_multi_needle(
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                position_1=pos_1,
                position_2=pos_2,
                seed=seed,
                example_idx=i,
            )
        elif task == "variable_tracking":
            yield gen_variable_tracking(
                tokenizer=tokenizer,
                target_tokens=target_tokens,
                seed=seed,
                example_idx=i,
            )
        else:
            raise ValueError(f"unknown task: {task}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="benchmark/configs/benchmark.yaml")
    parser.add_argument("--profile", choices=["dev", "full"], default="dev")
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text())

    n_examples = cfg["profiles"][args.profile]["examples_per_task_length"]
    tokenizer = _load_tokenizer(cfg["tokenizer_model_id"])
    positions = cfg["needle_positions"]
    seed = cfg["seed"]

    out_root = Path(args.out_root or f"benchmark/data/{args.profile}")
    out_root.mkdir(parents=True, exist_ok=True)

    for task in cfg["tasks"]:
        for target_tokens in cfg["context_lengths"]:
            out_path = out_root / f"{task}_{target_tokens}.jsonl"
            print(f"[gen] {task} @ {target_tokens} tokens -> {out_path}")
            with out_path.open("w") as f:
                for ex in _gen_for_task(
                    task, tokenizer, target_tokens, n_examples, seed, positions
                ):
                    f.write(json.dumps(ex) + "\n")

    print(f"[done] profile={args.profile} examples_per_task_length={n_examples}")


if __name__ == "__main__":
    main()
