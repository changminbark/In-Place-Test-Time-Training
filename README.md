# Adapter-Only In-Place Test-Time Training

Isolating the contribution of fast-weight target modules in frozen LLMs, on Gemma3-1B.

[Original In-Place TTT paper (Feng et al.)](https://arxiv.org/pdf/2604.06169)

## Motivation

The In-Place Test-Time Training (In-Place TTT) paper trains the base model and TTT modules jointly during continual pretraining (~20B tokens at 32K context on H800s). This leaves an open question: how much of the long-context gain comes from the **TTT adapter modules** (Conv1D, W_target) learning a useful next-token-prediction target, versus the **base model co-adapting** to tolerate the dynamic weight updates?

We isolate the first contribution by **freezing the base model and training only the TTT adapter modules**, then comparing against vanilla Gemma3 on RULER-style long-context tasks.

## Approach

Use `google/gemma-3-1b-it` (26 layers, hidden 1152, GeGLU MLPs, 32K context) as the base. We implement In-Place TTT as a drop-in enhancement: a `Conv1D` + `W_target` adapter is added to the MLP of selected layers, gated on `config.use_ttt`. Only the adapter parameters receive gradients during training; the base model is fully frozen.

We skip continual pretraining entirely and train only via long-context supervised finetuning (~100M–500M tokens at 4K–8K context, FineWeb-Edu).

### Evaluation (NVIDIA RULER protocol)

- **ICL baseline**: input text is prepended to the context. A fresh frozen model with the full context is loaded per question.
- **In-Place TTT**: the model processes the input text and updates its fast weights. A fresh frozen model + pretrained TTT modules (no input context in the prompt) is loaded per question, relying on weight-compressed knowledge.

Tested across context lengths from 1K to 32K tokens on RULER tasks (single/multi-hop NIAH, variable tracking, QA). Metrics: answer accuracy, GPU memory, inference latency.

## Repository layout

```
In-Place-Test-Time-Training/
├── models/
│   └── hf_gemma3/
│       ├── config_gemma3.py     # Gemma3TTTConfig: subclasses upstream Gemma3TextConfig, adds TTT fields
│       ├── model_gemma3.py      # Gemma3MLPTTT, Gemma3DecoderLayerTTT, Gemma3TextModelTTT, Gemma3ForCausalLMTTT
│       └── test_gemma3.py       # pytest suite: instantiation, forward, generate, save/load round-trip, freeze
├── train/
│   └── main.py                  # training entry point (frozen base + TTT-adapter SFT)
├── eval/
│   └── ruler.py                 # RULER evaluation harness
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
- `Gemma3TextModelTTT`, `Gemma3ForCausalLMTTT` — backbone + LM head. `freeze_base_model()` on the LM marks everything except `ttt_proj` + `ttt_conv` as `requires_grad=False`.

When `config.use_ttt=False`, the TTT branches are skipped entirely and the model behaves identically to upstream Gemma3.

## Setup

```bash
make install       # uv sync --all-groups
make test          # fast tests (skip slow ones)
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
model.freeze_base_model()            # only ttt_proj + ttt_conv get gradients
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

## Training

```bash
make train
# or directly:
uv run python train/main.py
```

Edit `train/main.py` to set TTT layer placement, training context length (4K–8K), batch size, learning rate, and dataset (FineWeb-Edu). The script:

1. Loads `google/gemma-3-1b-it` weights into a `Gemma3ForCausalLMTTT` with `use_ttt=True`.
2. Calls `freeze_base_model()` to gate gradients to TTT adapter parameters only.
3. Runs SFT on long-context documents.
4. Saves a checkpoint to `./checkpoints/gemma3-1b-ttt/`.

## Pushing to the HuggingFace Hub

After training:

```bash
make login-hf                                          # huggingface-cli login (one time)
make push-hub HF_REPO_ID=yourname/gemma3-1b-ttt CKPT_DIR=checkpoints/gemma3-1b-ttt
```

For Hub users to load the model with `AutoModelForCausalLM.from_pretrained(..., trust_remote_code=True)`, the repo must contain:

- `config.json` (written by `save_pretrained`), with an `auto_map` entry:
  ```json
  "auto_map": {
      "AutoConfig": "config_gemma3.Gemma3TTTConfig",
      "AutoModelForCausalLM": "model_gemma3.Gemma3ForCausalLMTTT"
  }
  ```
- `config_gemma3.py` and `model_gemma3.py` (the modeling code).
- `model.safetensors` (trained weights, including base + TTT adapter).
- A model card (`README.md`) noting:
  - License: `apache-2.0` (modeling code is yours, base weights are Gemma).
  - `base_model: google/gemma-3-1b-it`.
  - That base weights remain subject to the [Gemma Terms of Use](https://ai.google.dev/gemma/terms).
  - Acknowledgement of the In-Place TTT paper.

If `auto_map` isn't in `config.json`, copy `config_gemma3.py` and `model_gemma3.py` into the checkpoint folder before `push-hub` and add the `auto_map` block manually. Hub-side examples: see HuggingFace's [custom code documentation](https://huggingface.co/docs/transformers/custom_models).

## Evaluation

```bash
make eval
```

Runs the RULER-style protocol described above against vanilla Gemma3-1B (baseline) and Gemma3-1B + TTT adapter, reporting accuracy / memory / latency vs context length.

## Make targets

Run `make help` for the full list. Highlights:

| Target | Description |
| --- | --- |
| `make install` | `uv sync --all-groups` |
| `make test` | fast pytest suite (skips slow) |
| `make test-slow` | downloads real Gemma3-1B and exercises the load path |
| `make train` | runs `train/main.py` |
| `make eval` | runs `eval/ruler.py` |
| `make login-hf` | `huggingface-cli login` |
| `make push-hub` | upload `$(CKPT_DIR)` to `$(HF_REPO_ID)` |
| `make clean` | nuke `__pycache__`, `.pytest_cache`, etc. |

## Tech stack

PyTorch, HuggingFace Transformers, NVIDIA RULER, HuggingFace Datasets, Weights & Biases.

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