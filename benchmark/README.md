# Long-Context Retrieval Mini-Benchmark

Measures whether `Gemma 3 1B` (vanilla vs + TTT adapter) can retrieve facts from long synthetic documents. Inspired by RULER; smaller, synthetic, deterministic exact-match scoring.

## Tasks

- `single_needle` — retrieve one fact from a distractor document.
- `multi_needle` — retrieve and compose two linked facts.
- `variable_tracking` — return the final value of a target variable after many state updates.

## Modes

All three run on the same example set.

- `icl` — prompt = `[doc, q]`. Single forward. Vanilla baseline.
- `ttt_paper` — prompt = `[doc, q]`. Single forward; TTT layers update fast weights chunk-by-chunk during prefill, reset between examples. Matches the paper's RULER eval.
- `ttt_strict` — two-phase. (1) Ingest: forward over doc only, snapshot per-layer cumulative `ΔW`. (2) Answer: forward over `q` only with the snapshot patched in. Doc absent from answer prompt — fast weights must substitute for context, not aid it.

## Configuration

| | values |
|---|---|
| Context lengths (Gemma 3 tokens) | 1024, 4096, 8192, 16384, 32768 |
| Profiles | `dev` = 25 / `full` = 100 examples per task per length |
| Needle positions | `early`, `middle`, `late` (single/multi only) |
| Generation | `max_new_tokens=16`, greedy |
| Scoring | normalized exact match: lowercase, trim, strip surrounding punctuation |

## Data + result formats (JSONL)

Example: `id`, `task`, `context_length_target`, `document`, `question`, `answer`, `answer_aliases`, `metadata`.

Result row: `example_id`, `task`, `mode`, `model_name`, `context_length_target`, `prediction`, `ground_truth`, `correct`, `latency_ms`, `ingest_latency_ms`, `answer_latency_ms`, `peak_gpu_memory_mb`, `metadata`.

## Run

```bash
make install
echo 'HF_TOKEN=hf_xxx' > .env

uv run python -m benchmark.scripts.generate --profile dev
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_icl_factory
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_ttt_paper_factory
uv run python -m benchmark.scripts.evaluate --profile dev --predictor benchmark.eval.factories:gemma3_ttt_strict_factory
uv run python -m benchmark.scripts.aggregate
uv run python -m benchmark.scripts.report
uv run python -m benchmark.scripts.plot
```

Override checkpoints via `GEMMA3_BASE_MODEL_ID` / `GEMMA3_TTT_MODEL_ID` env vars.

Smoke test (no model, no HF download): `uv run python -m benchmark.scripts.smoke_test`.

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
  data_gen/                # synthetic generators
  eval/
    predictor.py           # Predictor / SinglePassPredictor / StrictTTTPredictor
    runner.py
    scoring.py
    factories.py           # echo_* + gemma3_*
    gemma3_predictors.py   # model loading + generate-injection wrapper
  scripts/                 # generate, evaluate, aggregate, report, plot, smoke_test
  results/                 # gitignored
```

## Validity

A run is valid iff every example yields one prediction, scoring is deterministic, all modes share the same example set, and `dev` / `full` share the schema.
