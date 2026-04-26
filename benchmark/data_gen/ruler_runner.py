"""Run NVIDIA/RULER's synthetic data generators as subprocesses and convert
the output to our benchmark JSONL schema.

RULER lives at `third_party/RULER` (git submodule). Its scripts are CLI-driven
(argparse + jsonl output via manifest_utils). We invoke them per (task, length,
seed), read the resulting jsonl, and remap fields:

  RULER row                     ours
  -----------------------       ------------------------------------
  input (full prompt text)      → split into `document` + `question`
  outputs (list[str])           → answer (first), answer_aliases
  length                        → metadata.token_length
  index, token_position_answer  → metadata
  + task variant from cfg       → metadata.task_variant

`document` and `question` are split at RULER's `answer_prefix` anchor (the
suffix added to the prompt right before the model would generate). This keeps
ICL/TTT-paper/TTT-strict behaviorally consistent with our predictor templates,
while preserving every token RULER chose to put in the prompt.
"""

from __future__ import annotations

import json
import re
import string
import subprocess
import tempfile
from pathlib import Path
from typing import Dict, Iterator, List


_REPO_ROOT = Path(__file__).resolve().parents[2]
_RULER_SYNTHETIC = _REPO_ROOT / "third_party" / "RULER" / "scripts" / "data" / "synthetic"


def _ruler_task_constants(task_key: str) -> Dict:
    """Load RULER's TASKS dict from third_party/RULER/.../constants.py without
    importing it (avoids polluting our argparse / sys.path). Each entry has
    keys: 'tokens_to_generate', 'template', 'answer_prefix'.
    """
    import importlib.util
    spec = importlib.util.spec_from_file_location(
        "_ruler_constants", _RULER_SYNTHETIC / "constants.py"
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.TASKS[task_key]


# ---------------------------------------------------------------------------
# Task configs — direct port of third_party/RULER/scripts/synthetic.yaml.
# QA tasks are deliberately omitted (they require external dataset downloads).
# ---------------------------------------------------------------------------

RULER_TASK_CONFIGS: Dict[str, Dict] = {
    "niah_single_1": {
        "script": "niah.py",
        "args": {"type_haystack": "noise", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_single_2": {
        "script": "niah.py",
        "args": {"type_haystack": "essay", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_single_3": {
        "script": "niah.py",
        "args": {"type_haystack": "essay", "type_needle_k": "words",
                 "type_needle_v": "uuids", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_multikey_1": {
        "script": "niah.py",
        "args": {"type_haystack": "essay", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 4, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_multikey_2": {
        "script": "niah.py",
        "args": {"type_haystack": "needle", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_multikey_3": {
        "script": "niah.py",
        "args": {"type_haystack": "needle", "type_needle_k": "uuids",
                 "type_needle_v": "uuids", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_multivalue": {
        "script": "niah.py",
        "args": {"type_haystack": "essay", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 4, "num_needle_q": 1},
        "tokens_to_generate": 128,
    },
    "niah_multiquery": {
        "script": "niah.py",
        "args": {"type_haystack": "essay", "type_needle_k": "words",
                 "type_needle_v": "numbers", "num_needle_k": 1, "num_needle_v": 1, "num_needle_q": 4},
        "tokens_to_generate": 128,
    },
    "vt": {
        "script": "variable_tracking.py",
        "args": {"type_haystack": "noise", "num_chains": 1, "num_hops": 4},
        "tokens_to_generate": 30,
    },
    "cwe": {
        "script": "common_words_extraction.py",
        "args": {"freq_cw": 30, "freq_ucw": 3, "num_cw": 10},
        "tokens_to_generate": 120,
    },
    "fwe": {
        "script": "freq_words_extraction.py",
        "args": {"alpha": 2.0},
        "tokens_to_generate": 50,
    },
}


# RULER's per-task structure puts the question on its own final line of the
# template (after the last `\n`), and writes the answer-prefix to a SEPARATE
# JSONL field (not concatenated into `input`). We split on the last newline
# and stitch the answer prefix back onto the question side.


_PUNCT_RE = re.compile(r"^[\s" + re.escape(string.punctuation) + r"]+|[\s" + re.escape(string.punctuation) + r"]+$")


def normalize_answer(
    text: str,
    lowercase: bool = True,
    strip_whitespace: bool = True,
    strip_punctuation: bool = True,
) -> str:
    out = text
    if lowercase:
        out = out.lower()
    if strip_whitespace:
        out = out.strip()
    if strip_punctuation:
        out = _PUNCT_RE.sub("", out)
    return out


_TASK_KEYS = {
    "niah.py": "niah",
    "variable_tracking.py": "variable_tracking",
    "common_words_extraction.py": "common_words_extraction",
    "freq_words_extraction.py": "freq_words_extraction",
}


def _build_argv(script: str, args: Dict, save_dir: Path, save_name: str,
                tokenizer_model_id: str, max_seq_length: int,
                num_samples: int, seed: int, tokens_to_generate: int) -> List[str]:
    # RULER's CLIs require --template to be passed explicitly. The canonical
    # templates live in constants.TASKS keyed by task family.
    task_const = _ruler_task_constants(_TASK_KEYS[script])
    template_str = task_const["template"] + task_const["answer_prefix"]

    argv = [
        "python", script,
        "--save_dir", str(save_dir),
        "--save_name", save_name,
        "--subset", "validation",
        "--tokenizer_path", tokenizer_model_id,
        "--tokenizer_type", "hf",
        "--max_seq_length", str(max_seq_length),
        "--tokens_to_generate", str(tokens_to_generate),
        "--num_samples", str(num_samples),
        "--random_seed", str(seed),
        "--template", template_str,
    ]
    for k, v in args.items():
        argv.extend([f"--{k}", str(v)])
    return argv


def _split_document_question(input_text: str, answer_prefix: str) -> tuple[str, str]:
    """Split RULER's `input` (instruction + context + question) on the last
    newline. The question line is the trailing segment; document is everything
    before. The answer_prefix (separate RULER field) is appended to the question
    so the final prompt is identical to RULER's full string.
    """
    nl = input_text.rfind("\n")
    if nl < 0:
        # No newline → treat the whole thing as the document; question is just
        # the answer prefix.
        return input_text.rstrip(), answer_prefix.lstrip()
    document = input_text[:nl].rstrip()
    question_line = input_text[nl + 1:].strip()
    question = (question_line + " " + answer_prefix.strip()).strip()
    return document, question


def _to_our_schema(row: Dict, task: str, target_tokens: int, idx: int, script: str) -> Dict:
    raw_input = row.get("input", "")
    answer_prefix = row.get("answer_prefix", "")
    document, question = _split_document_question(raw_input, answer_prefix)
    outputs = row.get("outputs") or row.get("output") or []
    if isinstance(outputs, str):
        outputs = [outputs]
    answer = outputs[0] if outputs else ""
    return {
        "id": f"{task}_{target_tokens}_{idx:04d}",
        "task": task,
        "context_length_target": target_tokens,
        "document": document,
        "question": question,
        "answer": answer,
        "answer_aliases": list(outputs),
        "metadata": {
            "task_variant": task,
            "ruler_length": row.get("length"),
            "ruler_token_position_answer": row.get("token_position_answer"),
            "ruler_index": row.get("index"),
        },
    }


def generate_examples(
    task: str,
    target_tokens: int,
    num_samples: int,
    tokenizer_model_id: str,
    seed: int,
) -> Iterator[Dict]:
    """Run RULER's CLI for one (task, length, seed) and yield our-schema rows."""
    if task not in RULER_TASK_CONFIGS:
        raise KeyError(f"unknown RULER task {task!r}; known: {sorted(RULER_TASK_CONFIGS)}")
    cfg = RULER_TASK_CONFIGS[task]

    if not _RULER_SYNTHETIC.exists():
        raise RuntimeError(
            f"RULER submodule missing at {_RULER_SYNTHETIC}. "
            "Run: git submodule update --init --recursive"
        )

    with tempfile.TemporaryDirectory() as tmp:
        tmp_path = Path(tmp)
        argv = _build_argv(
            script=cfg["script"],
            args=cfg["args"],
            save_dir=tmp_path,
            save_name=task,
            tokenizer_model_id=tokenizer_model_id,
            max_seq_length=target_tokens,
            num_samples=num_samples,
            seed=seed,
            tokens_to_generate=cfg["tokens_to_generate"],
        )
        result = subprocess.run(
            argv,
            cwd=_RULER_SYNTHETIC,
            capture_output=True,
            text=True,
        )
        if result.returncode != 0:
            raise RuntimeError(
                f"RULER {task}@{target_tokens} failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )

        out_file = tmp_path / task / "validation.jsonl"
        if not out_file.exists():
            raise RuntimeError(f"RULER did not produce {out_file}")

        with out_file.open() as f:
            for idx, line in enumerate(f):
                line = line.strip()
                if not line:
                    continue
                row = json.loads(line)
                yield _to_our_schema(row, task=task, target_tokens=target_tokens,
                                     idx=idx, script=cfg["script"])
