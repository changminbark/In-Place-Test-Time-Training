"""Train Gemma3TTT on a single dataset and push the checkpoint to the HF Hub.

Trainable params are determined by `Gemma3ForCausalLMTTT.freeze_base_model()`:
ttt_conv, ttt_proj (W_target), and the MLP down_proj (W_down). Everything else
in the base model stays frozen.

Pick the dataset with `--dataset`:
    --dataset tinystories   # roneneldan/TinyStories
    --dataset longalpaca    # Yukang/LongAlpaca-12k

Example:
    python -m train.main --dataset tinystories --hf-user my-hf-username
    python -m train.main --dataset longalpaca  --hf-user my-hf-username
"""
from __future__ import annotations

import argparse
import os
import shutil
from pathlib import Path
from typing import Optional

import torch
from datasets import load_dataset
from transformers import (
    AutoTokenizer,
    DataCollatorForLanguageModeling,
    Trainer,
    TrainingArguments,
)

from models.hf_gemma3.config_gemma3 import Gemma3TTTConfig
from models.hf_gemma3.model_gemma3 import Gemma3ForCausalLMTTT


MODEL_DIR = Path(__file__).resolve().parent.parent / "models" / "hf_gemma3"

AUTO_MAP = {
    "AutoConfig": "config_gemma3.Gemma3TTTConfig",
    "AutoModelForCausalLM": "model_gemma3.Gemma3ForCausalLMTTT",
}

DATASET_CHOICES = ("tinystories", "longalpaca")


# ---------------------------------------------------------------------------
# Dataset prep
# ---------------------------------------------------------------------------

def _tokenize(examples, tokenizer, text_key: str, max_length: int):
    # Don't pre-add `labels`: DataCollatorForLanguageModeling(mlm=False) clones
    # them from padded input_ids. Pre-adding them at un-padded length breaks
    # collation because the collator only pads input_ids/attention_mask.
    return tokenizer(
        examples[text_key],
        truncation=True,
        max_length=max_length,
        padding=False,
        return_attention_mask=True,
    )


def build_tinystories(tokenizer, max_length: int, max_samples: Optional[int]):
    ds = load_dataset("roneneldan/TinyStories", split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    return ds.map(
        lambda ex: _tokenize(ex, tokenizer, "text", max_length),
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenize TinyStories",
    )


def _format_longalpaca(example) -> str:
    instr = example.get("instruction", "") or ""
    inp = example.get("input", "") or ""
    out = example.get("output", "") or ""
    if inp:
        return f"### Instruction:\n{instr}\n\n### Input:\n{inp}\n\n### Response:\n{out}"
    return f"### Instruction:\n{instr}\n\n### Response:\n{out}"


def build_longalpaca(tokenizer, max_length: int, max_samples: Optional[int]):
    ds = load_dataset("Yukang/LongAlpaca-12k", split="train")
    if max_samples is not None:
        ds = ds.select(range(min(max_samples, len(ds))))
    ds = ds.map(
        lambda ex: {"text": _format_longalpaca(ex)},
        remove_columns=ds.column_names,
        desc="format LongAlpaca",
    )
    return ds.map(
        lambda ex: _tokenize(ex, tokenizer, "text", max_length),
        batched=True,
        remove_columns=ds.column_names,
        desc="tokenize LongAlpaca",
    )


def build_dataset(name: str, tokenizer, max_length: int, max_samples: Optional[int]):
    if name == "tinystories":
        return build_tinystories(tokenizer, max_length, max_samples)
    if name == "longalpaca":
        return build_longalpaca(tokenizer, max_length, max_samples)
    raise ValueError(f"unknown dataset: {name}")


# ---------------------------------------------------------------------------
# Save / push helpers
# ---------------------------------------------------------------------------

def bundle_remote_code(save_dir: Path) -> None:
    """Copy modeling files into the checkpoint dir so it can be loaded via
    `trust_remote_code=True` from the Hub."""
    for fname in ("config_gemma3.py", "model_gemma3.py", "__init__.py"):
        src = MODEL_DIR / fname
        if src.exists():
            shutil.copy(src, save_dir / fname)


def save_with_auto_map(
    model: Gemma3ForCausalLMTTT,
    tokenizer,
    save_dir: Path,
    repo_id: Optional[str],
    push_to_hub: bool,
) -> None:
    save_dir.mkdir(parents=True, exist_ok=True)
    model.config.auto_map = AUTO_MAP
    model.save_pretrained(save_dir)
    tokenizer.save_pretrained(save_dir)
    bundle_remote_code(save_dir)
    if push_to_hub and repo_id:
        print(f"[push] -> {repo_id}")
        model.push_to_hub(repo_id)
        tokenizer.push_to_hub(repo_id)
        # push_to_hub on the model alone does not include the .py files we
        # just copied next to the weights; upload them explicitly.
        from huggingface_hub import upload_file
        for fname in ("config_gemma3.py", "model_gemma3.py", "__init__.py"):
            fpath = save_dir / fname
            if fpath.exists():
                upload_file(
                    path_or_fileobj=str(fpath),
                    path_in_repo=fname,
                    repo_id=repo_id,
                    repo_type="model",
                )


# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------

def load_model_and_tokenizer(
    base_model: str,
    ttt_layers,
    ttt_chunk: int,
    ttt_lr: float,
    ttt_proj: bool,
    ttt_target: str,
):
    tokenizer = AutoTokenizer.from_pretrained(base_model)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    # Pass every TTT field explicitly: the base Gemma3 config.json has none of
    # them, and from_pretrained overlay-merging can leave them at None instead
    # of falling back to the class defaults.
    cfg = Gemma3TTTConfig.from_pretrained(
        base_model,
        use_ttt=True,
        ttt_layers=list(ttt_layers),
        ttt_chunk=ttt_chunk,
        ttt_lr=ttt_lr,
        ttt_proj=ttt_proj,
        ttt_target=ttt_target,
    )
    model = Gemma3ForCausalLMTTT.from_pretrained(
        base_model, config=cfg, torch_dtype=torch.bfloat16,
    )
    model.freeze_base_model()
    n_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[freeze] trainable params: {n_trainable:,}")
    return model, tokenizer


def _setup_wandb(
    *, enabled: bool, project: str, run_name: Optional[str], dataset: str,
) -> bool:
    """Configure wandb env vars before Trainer instantiates the integration.

    Returns True if wandb logging should be enabled. Falls back to disabled
    (with a printed note) if `wandb` isn't installed or no credentials are
    available outside an interactive shell.
    """
    if not enabled:
        os.environ["WANDB_DISABLED"] = "true"
        return False
    import importlib.util
    if importlib.util.find_spec("wandb") is None:
        print("[wandb] package not installed; skipping wandb logging")
        os.environ["WANDB_DISABLED"] = "true"
        return False
    os.environ.setdefault("WANDB_PROJECT", project)
    if run_name:
        os.environ["WANDB_NAME"] = run_name
    else:
        os.environ.setdefault("WANDB_NAME", f"gemma3-ttt-{dataset}")
    return True


def train_on_dataset(
    model: Gemma3ForCausalLMTTT,
    tokenizer,
    train_dataset,
    output_dir: Path,
    *,
    epochs: float,
    batch_size: int,
    grad_accum: int,
    lr: float,
    bf16: bool,
    save_steps: int,
    use_wandb: bool,
) -> None:
    collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)
    args = TrainingArguments(
        output_dir=str(output_dir / "trainer"),
        num_train_epochs=epochs,
        per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum,
        learning_rate=lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.03,
        weight_decay=0.0,
        logging_steps=10,
        save_steps=save_steps,
        save_total_limit=2,
        bf16=bf16,
        fp16=not bf16 and torch.cuda.is_available(),
        gradient_checkpointing=False,
        report_to=["wandb"] if use_wandb else "none",
        remove_unused_columns=False,
        dataloader_num_workers=2,
    )
    Trainer(
        model=model,
        args=args,
        train_dataset=train_dataset,
        data_collator=collator,
    ).train()


# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

# Per-dataset defaults tuned to each dataset's typical length distribution.
DATASET_DEFAULTS = {
    "tinystories": {
        "max_length": 1024,
        "batch_size": 4,
        "grad_accum": 4,
        "lr": 5e-5,
        "epochs": 1.0,
    },
    "longalpaca": {
        "max_length": 8192,
        "batch_size": 1,
        "grad_accum": 8,
        "lr": 5e-5,
        "epochs": 1.0,
    },
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True, choices=DATASET_CHOICES,
                   help="Which dataset to train on.")
    p.add_argument("--base-model", default="google/gemma-3-1b-it")
    p.add_argument("--hf-user", default=None,
                   help="HF username/org. If unset, no push is attempted.")
    p.add_argument("--output-dir", default="./checkpoints")
    p.add_argument("--ttt-layers", nargs="+", type=int,
                   default=[0, 6, 12, 18, 24])
    p.add_argument("--ttt-chunk", type=int, default=2048)
    p.add_argument("--ttt-lr", type=float, default=0.3,
                   help="Inner-loop η that scales the per-chunk ΔW.")
    p.add_argument("--ttt-proj", action=argparse.BooleanOptionalAction, default=True,
                   help="Enable the W_target (ttt_proj) module. "
                        "Use --no-ttt-proj to disable.")
    p.add_argument("--ttt-target", choices=("hidden_states", "input_embed"),
                   default="hidden_states",
                   help="Source for the TTT target stream.")

    p.add_argument("--epochs", type=float, default=None)
    p.add_argument("--batch-size", type=int, default=None)
    p.add_argument("--grad-accum", type=int, default=None)
    p.add_argument("--lr", type=float, default=None)
    p.add_argument("--max-length", type=int, default=None)
    p.add_argument("--max-samples", type=int, default=None)

    p.add_argument("--save-steps", type=int, default=500)
    p.add_argument("--no-bf16", action="store_true")
    p.add_argument("--no-push", action="store_true")
    p.add_argument("--repo-id", default=None,
                   help="Override the auto-generated HF repo id.")

    # wandb logging
    p.add_argument("--no-wandb", action="store_true",
                   help="Disable wandb logging.")
    p.add_argument("--wandb-project", default="gemma3-ttt")
    p.add_argument("--wandb-run-name", default=None,
                   help="Override the wandb run name (default: gemma3-ttt-<dataset>).")
    return p.parse_args()


def _resolve(arg_val, default_val):
    return default_val if arg_val is None else arg_val


def main():
    args = parse_args()
    defaults = DATASET_DEFAULTS[args.dataset]

    epochs = _resolve(args.epochs, defaults["epochs"])
    batch_size = _resolve(args.batch_size, defaults["batch_size"])
    grad_accum = _resolve(args.grad_accum, defaults["grad_accum"])
    lr = _resolve(args.lr, defaults["lr"])
    max_length = _resolve(args.max_length, defaults["max_length"])

    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)
    bf16 = not args.no_bf16
    push = (not args.no_push) and (args.hf_user is not None or args.repo_id is not None)

    print(f"=== Training Gemma3TTT on {args.dataset} ===")
    use_wandb = _setup_wandb(
        enabled=not args.no_wandb,
        project=args.wandb_project,
        run_name=args.wandb_run_name,
        dataset=args.dataset,
    )

    model, tokenizer = load_model_and_tokenizer(
        args.base_model,
        args.ttt_layers,
        args.ttt_chunk,
        args.ttt_lr,
        args.ttt_proj,
        args.ttt_target,
    )

    train_ds = build_dataset(args.dataset, tokenizer, max_length, args.max_samples)

    train_on_dataset(
        model, tokenizer, train_ds, output_dir,
        epochs=epochs,
        batch_size=batch_size,
        grad_accum=grad_accum,
        lr=lr,
        bf16=bf16,
        save_steps=args.save_steps,
        use_wandb=use_wandb,
    )

    if args.repo_id is not None:
        repo_id = args.repo_id
    elif args.hf_user is not None:
        base_tag = args.base_model.split("/")[-1].lower()
        repo_id = f"{args.hf_user}/{base_tag}-ttt-{args.dataset}"
    else:
        repo_id = None

    save_with_auto_map(model, tokenizer, output_dir / "final", repo_id, push)


if __name__ == "__main__":
    main()
