# Adapter-Only In-Place Test-Time Training

Isolating the contribution of fast-weight target modules in frozen LLMs, on Gemma3-1B.

[Original In-Place TTT paper (Feng et al.)](https://arxiv.org/pdf/2604.06169)

## Motivation

The In-Place Test-Time Training (In-Place TTT) paper trains the base model and TTT modules jointly during continual pretraining (~20B tokens at 32K context on H800s). This leaves an open question: how much of the long-context gain comes from the **TTT adapter modules** (Conv1D, W_target) learning a useful next-token-prediction target, versus the **base model co-adapting** to tolerate the dynamic weight updates?

We isolate the first contribution by **freezing the base model and training only the TTT adapter modules**, then comparing against vanilla Gemma3 on RULER-style long-context tasks.

## Approach

Use `google/gemma-3-1b-it` (26 layers, hidden 1152, GeGLU MLPs with intermediate 6912, 32K context) as the base. We implement In-Place TTT as a drop-in enhancement: a `Conv1D` + `W_target` adapter is added to the MLP of the global-attention layers `[0, 6, 12, 18, 24]`, gated on `config.use_ttt`.

We **skip continual pretraining entirely** and pretrain the adapter only. The base model is fully frozen except for the `down_proj` (W_down) of the TTT layers — that surface is the one the per-chunk ΔW updates, so it is trained jointly with `Conv1D` + `W_target`. Every other parameter (embeddings, attention, gate/up_proj, norms, lm_head, and the down_proj of all non-TTT layers) is `requires_grad=False`.

Primary training corpus: **500k samples of `roneneldan/TinyStories`** (short narratives, ~2M total). `Yukang/LongAlpaca-12k` is also supported as a long-context variant — note it caps at **12k samples**, so we run it for 2 epochs by default to compensate.

### Evaluation (NVIDIA RULER protocol)

Three configurations, each with a fresh frozen model loaded per question:

- **ICL baseline** — input text prepended to the context; vanilla Gemma3.
- **ICL + In-Place TTT** — input text prepended to the context; the model also has a pretrained TTT adapter.
- **In-Place TTT (no ICL)** — the model processes the input text and updates its fast weights; the question is asked with an empty context, relying entirely on the weight-compressed knowledge.

Tested across 1K / 4K / 8K / 16K / 32K context lengths on RULER tasks (single- and multi-hop NIAH, variable tracking, QA). Metrics: answer accuracy, GPU memory, inference latency, plus a needle-position × accuracy heatmap.

## Repository layout

```
In-Place-Test-Time-Training/
├── models/hf_gemma3/
│   ├── config_gemma3.py         # Gemma3TTTConfig: subclasses upstream Gemma3TextConfig, adds TTT fields
│   ├── model_gemma3.py          # Gemma3MLPTTT, Gemma3DecoderLayerTTT, Gemma3TextModelTTT, Gemma3ForCausalLMTTT
│   └── test_gemma3.py           # pytest suite: instantiation, forward, generate, save/load round-trip, freeze
├── train/
│   ├── main.py                  # training entry point (frozen base + TTT-adapter pretraining)
│   ├── test_main.py             # pytest suite: tokenize, freeze, save, wandb, CLI plumbing
│   └── README.md                # training details, default hyperparameters, Colab walkthrough
├── benchmark/                   # RULER-style evaluation harness (configs, data_gen, eval, scripts)
├── third_party/RULER/           # NVIDIA RULER as a git submodule
├── Makefile                     # convenience commands (see `make help`)
├── pyproject.toml               # deps managed by uv
├── LICENSE                      # Apache 2.0
└── NOTICE                       # attribution to HuggingFace, Google (Gemma), Bytedance (TTT reference)
```

### Modeling code, in detail

`model_gemma3.py` mirrors upstream `transformers.models.gemma3.modeling_gemma3` and adds:

- `TTTLinear`, `TTTConv1d` — marker subclasses of `nn.Linear` / `nn.Conv1d` so `_init_weights` can identify TTT modules unambiguously (avoids shape collisions with `q_proj`/`o_proj`).
- `Gemma3MLPTTT` — Gemma3 MLP with optional `ttt_proj` (W_target) + `ttt_conv` modules, chunked TTT update in `forward(x, t=...)`.
- `Gemma3DecoderLayerTTT` — Gemma3 decoder layer, near-mirror of upstream; only delta is a `target_states` kwarg threaded into `mlp(...)`.
- `Gemma3PreTrainedModelTTT` — inherits from upstream `Gemma3PreTrainedModel`. Custom `_init_weights` does diagonal init for `TTTLinear` (near-identity) and zero init for `TTTConv1d` (no-op start), and defers everything else to `super()` so `_is_hf_initialized` skip-flags are honored and loaded checkpoints aren't trampled.
- `Gemma3TextModelTTT`, `Gemma3ForCausalLMTTT` — backbone + LM head. `freeze_base_model()` on the LM keeps `ttt_proj` (W_target), `ttt_conv`, and the `down_proj` (W_down) of the **TTT layers only** trainable; every other parameter — including `down_proj` on non-TTT layers — is `requires_grad=False`.

When `config.use_ttt=False`, the TTT branches are skipped entirely and the model behaves identically to upstream Gemma3.

## Setup

```bash
make install       # uv sync --all-groups + RULER submodule + nltk data + PG-essay haystack
make test          # fast tests over models/ and train/ (skips @slow)
make test-slow     # downloads google/gemma-3-1b-it; needs HF auth + Gemma TOU acceptance
```

## Loading the model

### From scratch (random TTT init on top of Gemma3 base)

```python
from models.hf_gemma3 import Gemma3ForCausalLMTTT, Gemma3TTTConfig

config = Gemma3TTTConfig.from_pretrained(
    "google/gemma-3-1b-it",
    use_ttt=True,
    ttt_layers=[0, 6, 12, 18, 24],   # global layers in Gemma3-1B
    ttt_chunk=2048,
    ttt_lr=0.3,
)
model = Gemma3ForCausalLMTTT.from_pretrained("google/gemma-3-1b-it", config=config)
model.freeze_base_model()            # ttt_proj, ttt_conv, and down_proj on TTT layers get gradients
```

### From a trained checkpoint (local)

```python
model = Gemma3ForCausalLMTTT.from_pretrained("./checkpoints/gemma3-1b-ttt")
```

### From the HuggingFace Hub

```python
from transformers import AutoModelForCausalLM
model = AutoModelForCausalLM.from_pretrained(
    "yourname/gemma3-1b-ttt",
    trust_remote_code=True,          # custom modeling code lives in the Hub repo
)
```

`trust_remote_code=True` is required because `Gemma3ForCausalLMTTT` is not part of upstream `transformers`.

#### HF Repos
- https://huggingface.co/hungngo04/gemma-3-1b-it-ttt-tinystories-500k


## Training

`train/main.py` pretrains the TTT adapters (`ttt_conv`, `ttt_proj`/W_target) and the TTT-layer `down_proj` (W_down) on a single dataset selected via `--dataset`, then pushes the result to the Hub (bundled with the modeling code + `auto_map`).

```bash
# Primary run: 500k TinyStories samples
make train-tinystories HF_USER=<you>
# Long-context variant
make train-longalpaca  HF_USER=<you>
# or directly:
uv run python -m train.main --dataset tinystories --hf-user <you> --max-samples 500000
uv run python -m train.main --dataset longalpaca  --hf-user <you>
```

Supported datasets: `tinystories` (`roneneldan/TinyStories`) and `longalpaca` (`Yukang/LongAlpaca-12k`). See [`train/README.md`](train/README.md) for the full table of default hyperparameters, every CLI flag, wandb setup, and a Colab walkthrough.

## Pushing to the HuggingFace Hub

`train/main.py` handles this automatically: it sets `auto_map`, copies `config_gemma3.py` + `model_gemma3.py` next to the weights, and pushes to `<hf-user>/<base>-ttt-<dataset>` (override with `--repo-id`). Authenticate once with `make login-hf`. Use `--no-push` to skip the upload.

For manually-built checkpoints, `make push-hub HF_REPO_ID=... CKPT_DIR=...` is still available; the repo must contain `config.json` with an `auto_map` block, the two `.py` modeling files, weights, and ideally a model card noting the Gemma base license. See HuggingFace's [custom code documentation](https://huggingface.co/docs/transformers/custom_models) for the standard layout.

## Evaluation

```bash
make eval
```

Runs the RULER-style protocol described above against the three configurations — ICL baseline, ICL + TTT, and TTT-only — reporting accuracy, GPU memory, and inference latency as a function of context length, plus a needle-position × accuracy heatmap. The harness lives under `benchmark/` (configs, data_gen, eval, scripts) and uses NVIDIA RULER from `third_party/RULER/` as a submodule. See [`benchmark/README.md`](benchmark/README.md) for details.

## Make targets

Run `make help` for the full list. Highlights:

| Target | Description |
| --- | --- |
| `make install` | `uv sync --all-groups` |
| `make test` | fast pytest suite (skips slow) |
| `make test-slow` | downloads real Gemma3-1B and exercises the load path |
| `make train DATASET=...` | trains on `tinystories`/`longalpaca` and pushes (`HF_USER=...`) |
| `make train-tinystories` / `make train-longalpaca` | dataset-specific shortcuts |
| `make eval` | runs `eval/ruler.py` |
| `make login-hf` | `huggingface-cli login` |
| `make push-hub` | upload `$(CKPT_DIR)` to `$(HF_REPO_ID)` |
| `make clean` | nuke `__pycache__`, `.pytest_cache`, etc. |

## Tech stack

PyTorch, HuggingFace Transformers, NVIDIA RULER, HuggingFace Datasets, Weights & Biases.

## Research question

> Can the TTT adapter modules (Conv1D, W_target) be trained while keeping the base model frozen, and still recover some fraction of the long-context gains reported in the paper?

## Expected outcomes

- Adapter-enhanced model matches the baseline at short contexts (no damage to base capability).
- Improves over baseline at longer contexts, but by a smaller margin than the paper's fully-trained variant.
- The size of that gap quantifies how much of the paper's reported gains require base-model co-adaptation.
- A finding of no improvement (or degradation) is itself a meaningful negative result: it would say the base model's adaptation is load-bearing, not just the adapter's learned target.

## Licensing

Modeling code is Apache 2.0. See `LICENSE` and `NOTICE` for full attribution to HuggingFace Transformers (Apache 2.0), Google (Gemma 3 architecture and weights, subject to the Gemma Terms of Use), and the Bytedance In-Place TTT reference implementation (Apache 2.0).

## Class Information

Chang Min Bark and Hung Ngo

CSCI357 (Spring 2026) — AI with Neural Nets

Professor Brian King

April 21, 2026

## AI Usage
AI tools like Claude Code were used to write documentation and parts of the code like Makefiles and tests.