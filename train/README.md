# Training Gemma3TTT

Fine-tunes the in-place TTT adapters on top of a frozen Gemma3 base.

## What's trained

`Gemma3ForCausalLMTTT.freeze_base_model()` leaves only three groups of params trainable; everything else (embeddings, attention, gate/up_proj, norms, lm_head) is frozen:

| Param substring | Module                  | Role                          |
| --------------- | ----------------------- | ----------------------------- |
| `ttt_conv`      | depthwise `Conv1d`      | per-chunk hidden-state filter |
| `ttt_proj`      | `Linear` (W_target)     | target projection             |
| `down_proj`     | MLP output (W_down)     | initial value the TTT ΔW updates |

Adapters live only on the layers listed in `ttt_layers` (default `[0, 6, 12, 18, 24]`). Init is near-identity: conv zeroed, ttt_proj sparse-diagonal, so the first forward matches the base model.

## Datasets

| `--dataset`    | Source                       | Notes                                |
| -------------- | ---------------------------- | ------------------------------------ |
| `tinystories`  | `roneneldan/TinyStories`     | short narratives, large sample count |
| `longalpaca`   | `Yukang/LongAlpaca-12k`      | long instruction/response pairs      |

LongAlpaca is reformatted into `### Instruction / ### Input / ### Response` blocks before tokenisation. Both datasets train as plain causal-LM (`labels = input_ids`).

## Default hyperparameters

Tuned for *adapter pretraining* on a frozen 1B base. Live in `DATASET_DEFAULTS` in `train/main.py`; override any of them on the CLI.

| Hyperparameter         | TinyStories | LongAlpaca | Flag                |
| ---------------------- | ----------- | ---------- | ------------------- |
| `max_length`           | 1024        | 4096       | `--max-length`      |
| `epochs`               | 1.0         | 2.0        | `--epochs`          |
| `batch_size`           | 8           | 1          | `--batch-size`      |
| `grad_accum`           | 8           | 16         | `--grad-accum`      |
| **effective batch**    | 64          | 16         | —                   |
| `lr`                   | 1e-4        | 5e-5       | `--lr`              |
| `weight_decay`         | 0.1         | 0.1        | `--weight-decay`    |
| `warmup_steps`         | 500         | 100        | `--warmup-steps`    |
| `max_grad_norm`        | 1.0         | 1.0        | `--max-grad-norm`   |

Shared optimizer / schedule (not per-dataset): AdamW with `(β1, β2) = (0.9, 0.95)`, cosine LR decay, bf16 (fp16 fallback on non-bf16 GPUs), gradient checkpointing off.

TTT-module knobs (defaults match the In-Place TTT paper):

| Hyperparameter | Default              | Flag             |
| -------------- | -------------------- | ---------------- |
| `ttt_layers`   | `[0, 6, 12, 18, 24]` | `--ttt-layers`   |
| `ttt_chunk`    | 2048                 | `--ttt-chunk`    |
| `ttt_lr` (η)   | 0.3                  | `--ttt-lr`       |
| `ttt_proj`     | on                   | `--no-ttt-proj`  |
| `ttt_target`   | `hidden_states`      | `--ttt-target`   |

## Local usage

```bash
# from repo root
huggingface-cli login                       # for base model + push
wandb login                                 # optional; auto-detected via WANDB_API_KEY

python -m train.main --dataset tinystories --hf-user <hf-user>
python -m train.main --dataset longalpaca  --hf-user <hf-user>
```

Each run pushes to `<hf-user>/<base>-ttt-<dataset>` (override with `--repo-id`). The pushed repo bundles `config_gemma3.py` + `model_gemma3.py` and sets `auto_map`, so consumers can just:

```python
AutoModelForCausalLM.from_pretrained("<hf-user>/<base>-ttt-tinystories", trust_remote_code=True)
```

### Other flags

- `--base-model` (default `google/gemma-3-1b-it`)
- `--max-samples N` to cap the dataset size (e.g. for a Colab session)
- `--no-push` to train without uploading; `--repo-id` to override the auto repo name
- `--no-bf16` if your GPU lacks bfloat16 (falls back to fp16 on CUDA)
- `--logging-steps` (default 10), `--save-steps` (default 500)
- `--wandb-project` (default `gemma3-ttt`), `--wandb-run-name`, `--no-wandb`

See `python -m train.main --help` for the complete list.

## Logging (wandb)

Wandb logging is on by default. The script will silently fall back to console-only logging if `wandb` isn't installed or you've never logged in. To enable it:

```bash
pip install wandb           # already pulled in by this repo's deps
wandb login                 # one-time
```

Configure the run via flags or env vars:

| Flag                | Env var          | Default              |
| ------------------- | ---------------- | -------------------- |
| `--wandb-project`   | `WANDB_PROJECT`  | `gemma3-ttt`         |
| `--wandb-run-name`  | `WANDB_NAME`     | `gemma3-ttt-<dataset>` |
| `--no-wandb`        | `WANDB_DISABLED` | unset (logging on)   |

Trainer logs `train/loss`, `learning_rate`, and `epoch` every `--logging-steps` (10).

## Running on Colab

Shrink `--max-length` to 4096 and use `--max-samples` to cap runtime.

```python
# Cell 1 — clone + install
!git clone https://github.com/<you>/In-Place-Test-Time-Training.git
%cd In-Place-Test-Time-Training
!pip install -q -e .

# Cell 2 — auth
from huggingface_hub import login
login()                       # paste an HF token with write access for push
import wandb; wandb.login()   # optional, but on by default — pass --no-wandb to skip

# Cell 3 — accept Gemma TOU once at https://huggingface.co/google/gemma-3-1b-it

# Cell 4 — train + push (TinyStories)
!python -m train.main \
    --dataset tinystories \
    --hf-user <your-hf-user> \
    --max-samples 50000          # trim for a Colab session

# Cell 5 — train + push (LongAlpaca)
!python -m train.main \
    --dataset longalpaca \
    --hf-user <your-hf-user> \
    --max-length 4096            # easier on T4 memory
```

Tips:

- Mount Drive and pass `--output-dir /content/drive/MyDrive/ttt-ckpts` to keep checkpoints across sessions.
- Free Colab disconnects after ~12 h of idle; use `--save-steps` and rely on Trainer resume from `output_dir/trainer` if you need to continue.
- If you hit OOM on LongAlpaca, lower `--max-length` first, then `--batch-size 1 --grad-accum 16`.
