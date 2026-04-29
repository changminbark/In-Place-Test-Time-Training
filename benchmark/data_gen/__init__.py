from .ruler_runner import RULER_TASK_CONFIGS, normalize_answer
from .ruler_runner import generate_examples as _ruler_generate_examples
from .helmet_runner import HELMET_TASKS
from .helmet_runner import generate_examples as _helmet_generate_examples


def generate_examples(task, target_tokens, num_samples, tokenizer_model_id, seed):
    """Dispatch to the right runner by task name. HELMET tasks are prefixed
    with `helmet_`; everything else goes through RULER."""
    if task in HELMET_TASKS:
        yield from _helmet_generate_examples(
            task=task,
            target_tokens=target_tokens,
            num_samples=num_samples,
            tokenizer_model_id=tokenizer_model_id,
            seed=seed,
        )
        return
    yield from _ruler_generate_examples(
        task=task,
        target_tokens=target_tokens,
        num_samples=num_samples,
        tokenizer_model_id=tokenizer_model_id,
        seed=seed,
    )


__all__ = ["RULER_TASK_CONFIGS", "HELMET_TASKS", "generate_examples", "normalize_answer"]
