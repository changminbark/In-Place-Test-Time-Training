# coding=utf-8
# Copyright 2025 Google LLC and the HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# This file is a modified version of transformers/models/gemma3/modeling_gemma3.py.
# Modifications Copyright 2026 Chang Min Bark and Hung Ngo.
# Modifications add In-Place Test-Time Training (TTT) adapter modules
# (Conv1D + W_target) to the MLP and a frozen-base training mode.
from huggingface_hub.dataclasses import strict

from transformers.models.gemma3.configuration_gemma3 import Gemma3TextConfig
from transformers.utils import logging

logger = logging.get_logger(__name__)


@strict
class Gemma3TTTConfig(Gemma3TextConfig):
    """Gemma3 text config with In-Place Test-Time Training (TTT) adapter fields.

    Inherits all rope/sliding-window/layer-type plumbing from Gemma3TextConfig
    so the upstream model code (rotary embed, attention, mask creation) finds
    every attribute it expects. TTT-specific fields are added below.
    """

    model_type = "gemma3_text"

    # TTT-related parameters
    use_ttt: bool = False
    ttt_chunk: int | None = 8192
    ttt_layers: list[int] | tuple[int, ...] | None = (0, 6, 12, 18, 24)
    ttt_lr: float | None = 0.3
    ttt_proj: bool | None = True
    ttt_target: str | None = "hidden_states"

    def __post_init__(self, **kwargs):
        super().__post_init__(**kwargs)

        # TTT settings
        if self.use_ttt:
            if self.ttt_layers is not None:
                self.ttt_layers = list(self.ttt_layers)
            if self.ttt_target not in {"hidden_states", "input_embed"}:
                raise ValueError("ttt_target must be one of {'hidden_states', 'input_embed'}")
        else:
            self.ttt_chunk = None
            self.ttt_layers = None
            self.ttt_lr = None
            self.ttt_proj = None
            self.ttt_target = None


__all__ = ["Gemma3TTTConfig"]
