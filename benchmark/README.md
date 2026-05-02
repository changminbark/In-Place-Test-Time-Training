# Long-Context Retrieval Benchmark

Eval harness for `Gemma 3 1B` (vanilla vs + TTT adapter) on long-context retrieval. Tasks come from two sources, both vendored as git submodules under `third_party/`:

- **RULER** (`third_party/RULER`) — synthetic recall and aggregation. Generators run as subprocesses; their output is mapped onto our JSONL schema.
- **HELMET** (`third_party/HELMET`) — real-world long-context tasks. ICL is HF-datasets backed (no extra download). RAG needs a one-time tarball pull (~4 GB).

Predictors handle in-context / paper-style TTT / strict TTT modes uniformly across both sources.

## Tasks

### RULER (active set)

| Task | RULER config | Description |
|---|---|---|
| `vt` | noise, 1 chain × 4 hops | Variable-tracking |
| `cwe` | freq=30/3, top-10 | Common-words extraction |
| `fwe` | α=2.0 | Frequent-words extraction |

NIAH variants were dropped — synthetic-token retrieval is dominated by attention and doesn't probe the fast-weight contribution. The RULER recall slice is largely a duplicate of HELMET's recall category.

### HELMET (active set)

| Task | Source | Lengths supported |
|---|---|---|
| `helmet_trec_coarse` | CogComp/trec (6-class) | 8k, 16k, 32k |
| `helmet_banking77` | PolyAI/banking77 (77-class) | 8k, 16k, 32k |
| `helmet_nq` | KILT NQ + retrieved Wikipedia | 8k, 16k, 32k |
| `helmet_hotpotqa` | KILT HotpotQA + retrieved Wikipedia | 8k, 16k, 32k |

Other HELMET ICL families (`helmet_trec_fine`, `helmet_clinic150`, `helmet_nlu`) and RAG families (`helmet_triviaqa`, `helmet_popqa`) are wired up but not in the default `tasks` list — add to `benchmark.yaml` to enable.

## Modes

All three run on the same example set.

- `in_context` — prompt = `[doc, q]`. Single forward. Vanilla baseline. (ATTENTION)
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

# RULER's cwe task needs english_words.json (8.5 MB, stored via Git LFS).
# Install LFS once and pull, otherwise cwe generation crashes with a JSON
# decode error.
sudo apt-get install -y git-lfs && git lfs install
git -C third_party/RULER lfs pull
# Alternative if you can't install lfs:
# curl -L https://media.githubusercontent.com/media/NVIDIA/RULER/main/scripts/data/synthetic/json/english_words.json \
#   -o third_party/RULER/scripts/data/synthetic/json/english_words.json

# HELMET RAG only — ~4 GB, populates third_party/HELMET/data/kilt/
bash third_party/HELMET/scripts/download_data.sh
# (or set HELMET_DATA_DIR=/path/to/helmet/data and download there)
```

HELMET ICL pulls its underlying datasets (banking77, trec, etc.) on demand via the HF `datasets` cache; nothing extra to download up front.

## Run

```bash
uv run python -m benchmark.scripts.generate --profile dev
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_in_context_factory
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
    return SinglePassPredictor("my-model", "in_context", make_my_generate_fn(model, tok))
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
    helmet_runner.py       # in-process loader for HELMET ICL + RAG
  eval/
    predictor.py           # Predictor / SinglePassPredictor / StrictTTTPredictor
    runner.py
    scoring.py
    factories.py           # echo_* + gemma3_*
    gemma3_predictors.py   # model loading + generate-injection wrapper
  scripts/                 # generate, evaluate, aggregate, report, plot, smoke_test
  results/                 # gitignored
third_party/RULER/         # submodule — NVIDIA/RULER
third_party/HELMET/        # submodule — princeton-nlp/HELMET
                            # data/ subfolder is gitignored (download on demand)
```

## Validity

A run is valid iff every example yields one prediction, scoring is deterministic, all modes share the same example set, and `dev` / `full` share the schema.
