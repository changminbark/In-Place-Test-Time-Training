"""Dump the rendered prompt for every active task × mode.

For each task in the benchmark.yaml active list, generate one example and
render the prompt that would actually be fed to model.generate() under each
mode. Writes a markdown doc with truncated head/tail for long prompts.

Usage:
    uv run python scripts/dump_eval_prompts.py
    uv run python scripts/dump_eval_prompts.py --length 8192
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from benchmark.data_gen import generate_examples
from benchmark.eval.predictor import SinglePassPredictor, StrictTTTPredictor


# Pick a small, fast length when the task supports it; HELMET tasks only
# support 8k+, so for those we pull the smallest supported length.
_TASK_DEFAULT_LENGTHS = {
    "vt": 1024,
    "cwe": 1024,
    "fwe": 1024,
    "helmet_trec_coarse": 8192,
    "helmet_banking77": 8192,
    "helmet_nq": 8192,
    "helmet_hotpotqa": 8192,
    "helmet_triviaqa": 8192,
    "helmet_popqa": 8192,
    "helmet_trec_fine": 8192,
    "helmet_clinic150": 8192,
    "helmet_nlu": 8192,
}


def _trunc(text: str, head_chars: int = 600, tail_chars: int = 400) -> str:
    """Show head + tail; collapse middle with a marker giving the omitted size."""
    if len(text) <= head_chars + tail_chars + 50:
        return text
    omitted = len(text) - head_chars - tail_chars
    return (
        text[:head_chars]
        + f"\n\n... [{omitted:,} characters elided] ...\n\n"
        + text[-tail_chars:]
    )


def _render_prompts(example: dict) -> dict:
    """Return {mode: rendered_prompt} for in_context, ttt_paper, ttt_strict."""
    sp = SinglePassPredictor(
        model_name="dummy",
        mode="in_context",
        generate_fn=lambda *a, **k: ("", 0.0, None),
    )
    # in_context and ttt_paper share the prompt (only the underlying model
    # differs); render once.
    if example.get("prompt"):
        ic_prompt = example["prompt"]
    else:
        ic_prompt = sp.prompt_template.format(
            document=example["document"], question=example["question"]
        )

    # ttt_strict uses only the question (document not in answer prompt).
    sp_strict = StrictTTTPredictor(
        model_name="dummy",
        ingest_fn=lambda *a, **k: (None, 0.0, None),
        answer_fn=lambda *a, **k: ("", 0.0, None),
    )
    if example.get("strict_answer_prompt"):
        strict_prompt = example["strict_answer_prompt"]
    else:
        strict_prompt = sp_strict.prompt_template.format(question=example["question"])

    return {
        "in_context_or_ttt_paper": ic_prompt,
        "ttt_strict_answer": strict_prompt,
        "ttt_strict_ingest": example["document"],  # what gets fed to ingest_fn
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="benchmark/configs/benchmark.yaml")
    ap.add_argument("--out", default="benchmark/results/arch/eval_prompts.md")
    ap.add_argument("--head-chars", type=int, default=800)
    ap.add_argument("--tail-chars", type=int, default=500)
    args = ap.parse_args()

    cfg = yaml.safe_load(Path(args.config).read_text())
    tasks = cfg["tasks"]
    tokenizer_id = cfg["tokenizer_model_id"]
    seed = cfg["seed"]

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    sections = ["# Eval prompt templates per task", ""]
    sections.append(
        "Each section shows the rendered prompt that would be passed to "
        "`model.generate()` for one sampled example, in each mode. Long bodies "
        "are truncated with a head/tail view; the elision marker shows how "
        "many characters are skipped."
    )
    sections.append("")

    for task in tasks:
        length = _TASK_DEFAULT_LENGTHS.get(task)
        if length is None:
            print(f"[skip] no default length for {task}")
            continue
        try:
            examples = list(generate_examples(
                task=task,
                target_tokens=length,
                num_samples=1,
                tokenizer_model_id=tokenizer_id,
                seed=seed,
            ))
        except (KeyError, FileNotFoundError) as e:
            print(f"[skip] {task}: {type(e).__name__}: {str(e).splitlines()[0]}")
            continue
        if not examples:
            print(f"[skip] {task}: no example produced")
            continue
        ex = examples[0]
        prompts = _render_prompts(ex)

        sections.append(f"## `{task}`  @  {length} tokens")
        sections.append("")
        sections.append(f"- **Gold answer**: `{ex['answer']!r}`")
        if ex.get("answer_aliases") and ex["answer_aliases"] != [ex["answer"]]:
            sections.append(f"- **Aliases** ({len(ex['answer_aliases'])}): "
                            f"`{ex['answer_aliases'][:3]!r}` …")
        sections.append(f"- **Prompt provenance**: "
                        f"{'HELMET-rendered (`example.prompt` field)' if ex.get('prompt') else 'predictor wrapper (`SinglePassPredictor.prompt_template`)'}")
        sections.append("")

        sections.append("### in_context / ttt_paper prompt")
        sections.append("")
        sections.append("```")
        sections.append(_trunc(prompts["in_context_or_ttt_paper"],
                               args.head_chars, args.tail_chars))
        sections.append("```")
        sections.append("")

        sections.append("### ttt_strict — phase 1 (ingest, document only)")
        sections.append("")
        sections.append("```")
        sections.append(_trunc(prompts["ttt_strict_ingest"],
                               args.head_chars, args.tail_chars))
        sections.append("```")
        sections.append("")

        sections.append("### ttt_strict — phase 2 (answer, question only)")
        sections.append("")
        sections.append("```")
        sections.append(prompts["ttt_strict_answer"])
        sections.append("```")
        sections.append("")

        full_len = len(prompts["in_context_or_ttt_paper"])
        print(f"[ok] {task:24s} @ {length:6d}  prompt_chars={full_len:,}")

    out_path.write_text("\n".join(sections))
    print(f"[done] wrote {out_path}")


if __name__ == "__main__":
    main()
