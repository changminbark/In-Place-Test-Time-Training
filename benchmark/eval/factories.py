"""Predictor factories.

A factory is a callable(config: dict) -> Predictor. The evaluate CLI accepts a
`module:function` spec and calls it with the loaded benchmark config.

Reference factories here are stubs (EchoPredictor) that let you run the whole
harness end-to-end without a real model. Real factories wired to Gemma 3 1B
will live alongside the model code (or in a separate module) once that lands.
"""

from __future__ import annotations

from .predictor import EchoPredictor


def echo_icl_factory(cfg: dict):
    return EchoPredictor(model_name="echo", mode="icl")


def echo_ttt_paper_factory(cfg: dict):
    return EchoPredictor(model_name="echo", mode="ttt_paper")


def echo_ttt_strict_factory(cfg: dict):
    return EchoPredictor(model_name="echo", mode="ttt_strict")


# Lazy re-exports of the real Gemma3-backed factories. Importing here would
# pull in torch/transformers/models.* even when only the echo factories are
# needed; instead, expose names that import on first call.

def gemma3_icl_factory(cfg: dict):
    from .gemma3_predictors import gemma3_icl_factory as _f
    return _f(cfg)


def gemma3_ttt_paper_factory(cfg: dict):
    from .gemma3_predictors import gemma3_ttt_paper_factory as _f
    return _f(cfg)


def gemma3_ttt_strict_factory(cfg: dict):
    from .gemma3_predictors import gemma3_ttt_strict_factory as _f
    return _f(cfg)
