# Long-Context Retrieval Benchmark

Eval harness for `Gemma 3 1B` (vanilla vs + TTT adapter) on long-context retrieval. Tasks are produced by NVIDIA/RULER (vendored as a git submodule under `third_party/RULER`); their generators run as subprocesses, output is mapped onto our JSONL schema, and our predictor abstraction handles ICL / paper-style TTT / strict TTT modes uniformly.

## Tasks (RULER variants)

| Task | RULER config | Description |
|---|---|---|
| `niah_single_1` | noise / words / numbers | Single needle in repetitive noise haystack |
| `niah_single_2` | essay / words / numbers | Single needle in PG essays |
| `niah_single_3` | essay / words / uuids | Single needle, UUID values (harder) |
| `niah_multikey_1` | essay, k=4 | One of four keys queried |
| `niah_multikey_2` | needle haystack, k=1 | Distractor needles surround the real one |
| `niah_multikey_3` | needle haystack, k=1, uuids | UUID distractors and target |
| `niah_multivalue` | essay, v=4 | One key, four values, retrieve all |
| `niah_multiquery` | essay, q=4 | Four queries against four keys |
| `vt` | noise, 1 chain × 4 hops | Variable-tracking |
| `cwe` | freq=30/3, top-10 | Common-words extraction |
| `fwe` | α=2.0 | Frequent-words extraction |

QA tasks (`qa_1` / `qa_2`) are not included — they require external dataset downloads.

## Modes

All three run on the same example set.

- `icl` — prompt = `[doc, q]`. Single forward. Vanilla baseline. (ATTENTION)
- `ttt_paper` — prompt = `[doc, q]`. Single forward; TTT layers update fast weights chunk-by-chunk during prefill, reset between examples. Matches the paper's RULER eval. (ATTENTION + TTT)
- `ttt_strict` — two-phase. (1) Ingest: forward over doc only, snapshot per-layer cumulative `ΔW`. (2) Answer: forward over `q` only with the snapshot patched in. Doc absent from answer prompt — fast weights must substitute for context, not aid it. (TTT)

## Configuration

| | values |
|---|---|
| Context lengths (Gemma 3 tokens) | 1024, 4096, 8192, 16384, 32768 |
| Profiles | `dev` = 25 / `full` = 100 examples per task per length |
| Generation | `max_new_tokens=16`, greedy |
| Scoring | normalized exact match: lowercase, trim, strip surrounding punctuation |

## Data + result formats (JSONL)

Example: `id`, `task`, `context_length_target`, `document`, `question`, `answer`, `answer_aliases`, `metadata`.

`document` and `question` are split from RULER's `input` at the answer-prefix anchor — `document` is everything up to the question line; `question` is the question + RULER's answer prefix.

Result row: `example_id`, `task`, `mode`, `model_name`, `context_length_target`, `prediction`, `ground_truth`, `correct`, `latency_ms`, `ingest_latency_ms`, `answer_latency_ms`, `peak_gpu_memory_mb`, `metadata`.

## One-time setup

```bash
make install                              # uv sync + submodule + nltk + PG essays
echo 'HF_TOKEN=hf_xxx' > .env             # accept Gemma license at HF first
```

## Run

```bash
uv run python -m benchmark.scripts.generate --profile dev
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_icl_factory
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_ttt_paper_factory
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_ttt_strict_factory
uv run python -m benchmark.scripts.aggregate
uv run python -m benchmark.scripts.report
uv run python -m benchmark.scripts.plot
```

Override checkpoints via `GEMMA3_BASE_MODEL_ID` / `GEMMA3_TTT_MODEL_ID` env vars.

Smoke test (one task, 3 examples, no real model needed): `uv run python -m benchmark.scripts.smoke_test`.

## Adding a predictor

```python
from benchmark.eval.predictor import SinglePassPredictor, StrictTTTPredictor

def my_factory(cfg):
    model, tok = load_my_model(...)
    return SinglePassPredictor("my-model", "icl", make_my_generate_fn(model, tok))
```

Run with `--predictor my_module:my_factory`.

## Layout

```
benchmark/
  spec.md
  configs/benchmark.yaml
  data/{dev,full}/         # gitignored
  data_gen/
    ruler_runner.py        # subprocess wrapper around RULER generators
  eval/
    predictor.py           # Predictor / SinglePassPredictor / StrictTTTPredictor
    runner.py
    scoring.py
    factories.py           # echo_* + gemma3_*
    gemma3_predictors.py   # model loading + generate-injection wrapper
  scripts/                 # generate, evaluate, aggregate, report, plot, smoke_test
  results/                 # gitignored
third_party/RULER/         # submodule — NVIDIA/RULER
```

## Validity

A run is valid iff every example yields one prediction, scoring is deterministic, all modes share the same example set, and `dev` / `full` share the schema.
